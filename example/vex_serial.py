import sys
import glob
import serial
import time
from multiprocessing import Process, Queue, Array, RawValue, Value
from ctypes import c_double, c_bool, c_int
import socket 
import json
import copy

ROTATION_DIRECTION = {
    "counter_clockwise": 1,
    "clockwise": -1
}

MINIMUM_INPLACE_ROTATION_SPEED = 60

from enum import Enum
class ARM_POSITION(Enum):
  low = 'low',
  mid = 'mid'
  high = 'high'


class IndexableArray:
  def __init__(self, length):
    self._data = Array(c_int, length)

  def __len__(self):
    return len(self._data)

  def __setitem__(self, idx, val):
    self._data.acquire()
    self._data[idx] = val
    self._data.release()

  def set(self, arr):
    self._data.acquire()
    self._data[:len(arr)] = arr
    self._data.release()

  def __iter__(self):
    return iter(self._data)

  def __getitem__(self, idx):
    if len(self._data) <= idx:
      raise ValueError("Index exceeds size of list")
    self._data.acquire()
    val = self._data[idx]
    self._data.release()
    return val

  def clone(self):
    self._data.acquire()
    _new_copy = self._data[:]
    self._data.release()
    return _new_copy

  def __str__(self):
    return str(self._data[:])

CMD_CONTROL_MOTOR_VALUES  = 'M'
CMD_STATUS_SENSOR_VALUES  = 'S'
CMD_STATUS_DEBUG          = 'I'
MLIMIT = 127

def _decode_message(msg):
  if len(msg) == 0: return None
  if msg[0] != '[' or msg[-1] != ']': return None

  if msg[1] not in [CMD_STATUS_SENSOR_VALUES]:#, CMD_STATUS_DEBUG]:
    return None

  sensor_values = []
  try:
    length = int(msg[2:4], base=16)
    if length != len(msg): return None

    chk_sum = int(msg[-3:-1], base=16)
    for c in msg[:-3]:
      chk_sum ^= ord(c)
    chk_sum ^= ord(msg[-1])
    if chk_sum != 0: return None

    ptr = 4
    while ptr < length - 3:
      _type = msg[ptr]
      if _type == 'w':
        nbytes = 1
      elif _type == 's':
        nbytes = 4
      elif _type == 'l':
        nbytes = 8
      else:
        nbytes = 1
      ptr += 1

      data = int(msg[ptr:ptr+nbytes], base=16)
      # detect negative values by looking at first bit in first byte
      if int(msg[ptr], base=16) & 0x8:
        max_byte_value = (1 << (nbytes << 2))
        data -= max_byte_value
      sensor_values.append(data)
      ptr += nbytes

    return sensor_values

  except ValueError:
    # print("Error: could not decode message, incorrect hexstring read")
    return None

def _receive_data(connection, rxbuf):
  msg = None
  buf = connection.read_all()
  if buf:
    try:
      rxbuf += buf.decode()
      end = rxbuf.find(']')
    except UnicodeDecodeError as e:
        return None, rxbuf
    if end != -1:
      start = rxbuf.find('[', 0, end)
      if start != -1 and '[' not in rxbuf[start+1:end]:
        msg =  rxbuf[start:end+1]
      rxbuf = rxbuf[end+1:]
  return msg, rxbuf

def _send_message(connection, vals):
  motor_correction = lambda x: \
    (-MLIMIT if x < -MLIMIT else (MLIMIT if x > MLIMIT else x)) + MLIMIT
  valstr = "".join(["%02x" % motor_correction(int(v)) for v in vals])
  length = len(valstr) + 7
  msg = "[M%02x%s]" % (length, valstr)
  chk_sum = 0
  for c in msg:
    chk_sum ^= ord(c)
  msg = msg[:-1] + ("%02x" % chk_sum) + "]\n"
  connection.write(msg.encode())

def _serial_worker(path, baud, motors, sensors, nsensors, enabled, readtime, keep_running):
  connection = serial.Serial(path, baud)
  rxbuf = ""
  last_tx_time = 0.0
  while keep_running.value:
    rx, rxbuf = _receive_data(connection, rxbuf)
    if rx:
      values = _decode_message(rx)
      if values:
        sensors._data.acquire()
        nsensors.value = len(values)
        sensors._data[:len(values)] = values
        sensors._data.release()
        readtime.acquire()
        readtime.value = time.time()
        readtime.release()

    # outgoing 50hz
    t = time.time()
    if t - last_tx_time > 0.02:
      last_tx_time = t
      values = [0] * len(motors)
      if enabled.value:
        values = motors.clone()
      _send_message(connection, values)

    # time.sleep(0.005) # throttle to prevent CPU overload
  
  motors.set([0] * len(motors))
  for _ in range(10):
    _send_message(connection, [0] * len(motors))
    time.sleep(0.05)
  connection.close()

class VexCortex:
  _entity = None

  def __init__(self, path=None, baud=115200):
    if VexCortex._entity is None:
      VexCortex._entity = self
    else:
      raise Exception("Already created VexCortex")

    self.baud = baud
    self._enabled = Value(c_bool, True)
    self._keep_running = RawValue(c_bool, True)

    self._sensor_values = IndexableArray(20)
    self._num_sensors = Value(c_int, 0)
    self._last_rx_time = Value(c_double, 0.0)
    self._motor_values = IndexableArray(10)
    self._worker = None

    self.path = path
    if self.path:
      try:
        s = serial.Serial(self.path, self.baud)
        s.close()
      except (OSError, serial.SerialException):
        raise EnvironmentError(f"Could not find specified path: {self.path}")
    else:
      self._autofind_path()
      if not self.path:
        raise EnvironmentError(f"Could not find any path")

    self._worker = Process(target=_serial_worker, args=(
      self.path, self.baud, self._motor_values, self._sensor_values,
      self._num_sensors, self._enabled, self._last_rx_time, self._keep_running))
    self._worker.start()

  def stop(self):
    self._keep_running.value = False
    if self._worker:
      self._worker.join(3)
      if self._worker.is_alive():
        self._worker.kill()
      self._worker = None

  def __del__(self):
    self.stop()

  def _autofind_path(self): # todo: do a more dynamic path finder using prefix
    # https://stackoverflow.com/questions/12090503/listing-available-com-ports-with-python?msclkid=bafb28c0ceb211ec97c565cfa73ea467
    if sys.platform.startswith('win'):
      paths = ["COM%s" % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux'):
      paths = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    else:
      raise EnvironmentError("Unsupported platform")

    if len(paths) == 0:
      raise EnvironmentError("Cannot find suitable port")

    for path in paths:
      try:
        s = serial.Serial(path, self.baud)
        s.close()
        self.path = path
        break # once it has found one, we are good
      except (OSError, serial.SerialException):
        self.path = None

  def enabled(self):
    return not self._enabled.value
  
  def running(self):
    return self._keep_running.value

  def timestamp(self):
    self._last_rx_time.acquire()
    timestamp = self._last_rx_time.value
    self._last_rx_time.release()
    return timestamp

  @property
  def motor(self):
    """Reference to the motor array

    Returns:
        IndexableArray: reference to the motor values
    """
    return self._motor_values

  def motors(self, motor_values):
    """Set motor values

    Args:
        motor_values (List[int]): motor values
    """
    self._motor_values.set(motor_values)

  @property
  def sensor(self):
    """Reference to the sensor array

    Returns:
        IndexableArray: reference to the sensor values
    """
    return self._sensor_values

  def sensors(self):
    """Get the sensor values

    Returns:
        List[int]: sensor values
    """
    self._sensor_values._data.acquire()
    num_sensors = self._num_sensors.value
    sensor_values = self._sensor_values._data[:num_sensors]
    self._sensor_values._data.release()
    return sensor_values
  
  def sensors_external(self):
    """Get the sensor values

    Returns:
        List[int]: sensor values
    """
    self._sensor_values._data.acquire()
    num_sensors = self._num_sensors.value
    sensor_values = self._sensor_values._data[:num_sensors]
    self._sensor_values._data.release()
    #make a deep copy of the sensor values to return 
    return_sensor_values = []
    for sensor_value in sensor_values:
      return_sensor_values.append(sensor_value)
    return return_sensor_values


import time
import numpy as np

class clawAction:
    Close = "close"
    Open = "open"
    Stop = "stop"


class VexControl:
    def __init__(self, robot):
        self.robot = robot
        self.angle = 90
        self.x = 0
        self.y = 0
        self.next_x = 0
        self.next_y = 0
        self.next_angle = 0
        self.prev_dist_error = 0
        self.dist_integral = 0
        self.prev_theta_error = 0
        self.theta_integral = 0
        self.search_direction = "clockwise"
    
    def catch_ball(self):
      #  self.claw(20, 'open',)
       self.drive(('forward', 30, 0.8))
       self.claw(20, 'close', 1, 0.8)
       self.drive('backward', 30, 0.2)
       self.update_robot_move_arm(armPosition=ARM_POSITION.high)


    def printQueueMessage(self, message):
        print(message)
    def setSearchDirection(self, direction):
        # assert if direction is not clockwise or counter_clockwise
        assert  direction == "clockwise" or direction == "counter_clockwise"
        self.search_direction = direction

    def send_to_XYTheta(self, target_x, target_y, target_theta, left_motor=0, right_motor=9):
        disable_Kdist = False
        if disable_Kdist:
            Kp_dist = 0.0
            Ki_dist = 0.0
            Kd_dist = 0.0
        else:
            Kp_dist = 6
            Ki_dist = 0.01
            Kd_dist = 0.00

        Kp_theta = 10
        Ki_theta = 0.01
        Kd_theta = 0.0

        # Compute distance and angle to target
        dx = target_x - self.x
        dy = target_y - self.y
        distance_error = np.sqrt(dx**2 + dy**2)
        target_angle = np.degrees(np.arctan2(dy, dx))
        angle_to_target = target_angle - self.angle
        angle_to_target = (angle_to_target + 180) % 360 - 180  # Normalize to [-180, 180]

        # Compute orientation error to target theta
        orientation_error = target_theta - self.angle
        orientation_error = (orientation_error + 180) % 360 - 180  # Normalize to [-180, 180]

        print(f'dx = {dx}, dy = {dy}')
        print(f'distance_error = {distance_error}, target_angle = {target_angle}, angle_to_target = {angle_to_target}')
        print(f'target_theta = {target_theta}, orientation_error = {orientation_error}')

        # PID calculations for distance
        self.dist_integral += distance_error
        dist_derivative = distance_error - self.prev_dist_error
        self.prev_dist_error = distance_error

        P_dist = Kp_dist * distance_error
        I_dist = Ki_dist * self.dist_integral
        D_dist = Kd_dist * dist_derivative

        print(f'P_dist = {P_dist}, I_dist = {I_dist}, D_dist = {D_dist}')
        print(f'dist_integral = {self.dist_integral}, dist_derivative = {dist_derivative}')

        # PID calculations for angle to target
        self.angle_integral += angle_to_target
        angle_derivative = angle_to_target - self.prev_angle_error
        self.prev_angle_error = angle_to_target

        P_angle = Kp_theta * angle_to_target
        I_angle = Ki_theta * self.angle_integral
        D_angle = Kd_theta * angle_derivative

        print(f'P_angle = {P_angle}, I_angle = {I_angle}, D_angle = {D_angle}')
        print(f'angle_integral = {self.angle_integral}, angle_derivative = {angle_derivative}')

        # PID calculations for orientation
        self.orientation_integral += orientation_error
        orientation_derivative = orientation_error - self.prev_orientation_error
        self.prev_orientation_error = orientation_error

        P_orientation = Kp_theta * orientation_error
        I_orientation = Ki_theta * self.orientation_integral
        D_orientation = Kd_theta * orientation_derivative

        print(f'P_orientation = {P_orientation}, I_orientation = {I_orientation}, D_orientation = {D_orientation}')
        print(f'orientation_integral = {self.orientation_integral}, orientation_derivative = {orientation_derivative}')

        # Compute motor outputs
        left_motor_speed = P_dist + I_dist + D_dist + P_angle + I_angle + D_angle + P_orientation + I_orientation + D_orientation
        right_motor_speed = P_dist + I_dist + D_dist - (P_angle + I_angle + D_angle + P_orientation + I_orientation + D_orientation)

        print(f'left_motor_speed (before clip) = {left_motor_speed}, right_motor_speed (before clip) = {right_motor_speed}')

        # Clip motor speeds to acceptable range
        left_motor_speed = np.clip(left_motor_speed, -127, 127)
        right_motor_speed = np.clip(right_motor_speed, -127, 127)

        print(f'left_motor_speed (after clip) = {left_motor_speed}, right_motor_speed (after clip) = {right_motor_speed}')

        # Set motor speeds
        motor_values = self.robot.motor
        motor_values[left_motor] = int(left_motor_speed)
        motor_values[right_motor] = int(right_motor_speed)

        print(f'motor_values[{left_motor}] = {motor_values[left_motor]}, motor_values[{right_motor}] = {motor_values[right_motor]}')

        # Simulate sending motor values to robot (uncomment when integrating with real robot)
        self.robot.motors(motor_values)

        # Sleep for a short duration to simulate control loop timing
        time.sleep(0.03)  # Adjust the sleep time as needed

        # Stop the robot
        self.stop_drive()

    def send_to_XY(self, target_x, target_y, left_motor=0, right_motor=9):
        disable_Kdist = False
        if disable_Kdist:
          Kp_dist = 0.0
          Ki_dist = 0.0
          Kd_dist = 0.000  # 0.1
        else:
          Kp_dist = 6
          Ki_dist = 0.01
          Kd_dist = 0.00  # 0.1
        Kp_theta = 10
        Ki_theta = 0.01
        Kd_theta = 0.000  # 0.1

        # Compute distance and angle to target
        dx = target_x - self.x
        dy = target_y - self.y
        distance_error = np.sqrt(dx**2 + dy**2)
        target_angle = np.degrees(np.arctan2(dy, dx))
        angle_error = target_angle - self.angle
        angle_error = (angle_error + 180) % 360 - 180  # Normalize to [-180, 180]

        print(f'dx = {dx}, dy = {dy}')
        print(f'distance_error = {distance_error}, target_angle = {target_angle}, angle_error = {angle_error}')

        # PID calculations for distance
        self.dist_integral += distance_error
        dist_derivative = distance_error - self.prev_dist_error
        self.prev_dist_error = distance_error

        P_dist = Kp_dist * distance_error
        I_dist = Ki_dist * self.dist_integral
        D_dist = Kd_dist * dist_derivative

        print(f'P_dist = {P_dist}, I_dist = {I_dist}, D_dist = {D_dist}')
        print(f'dist_integral = {self.dist_integral}, dist_derivative = {dist_derivative}')

        # PID calculations for angle
        self.theta_integral += angle_error
        theta_derivative = angle_error - self.prev_theta_error
        self.prev_theta_error = angle_error

        P_theta = Kp_theta * angle_error
        I_theta = Ki_theta * self.theta_integral
        D_theta = Kd_theta * theta_derivative

        print(f'P_theta = {P_theta}, I_theta = {I_theta}, D_theta = {D_theta}')
        print(f'theta_integral = {self.theta_integral}, theta_derivative = {theta_derivative}')

        # Compute motor outputs
        left_motor_speed = P_dist + I_dist + D_dist + P_theta + I_theta + D_theta
        right_motor_speed = P_dist + I_dist + D_dist - (P_theta + I_theta + D_theta)

        print(f'left_motor_speed (before clip) = {left_motor_speed}, right_motor_speed (before clip) = {right_motor_speed}')

        # Clip motor speeds to acceptable range
        left_motor_speed = np.clip(left_motor_speed, -127, 127)
        right_motor_speed = -np.clip(right_motor_speed, -127, 127)

        print(f'left_motor_speed (after clip) = {left_motor_speed}, right_motor_speed (after clip) = {right_motor_speed}')

        # Set motor speeds
        motor_values = self.robot.motor
        motor_values[left_motor] = int(left_motor_speed)
        motor_values[right_motor] = int(right_motor_speed)

        print(f'motor_values[{left_motor}] = {motor_values[left_motor]}, motor_values[{right_motor}] = {motor_values[right_motor]}')

        # Simulate sending motor values to robot (uncomment when integrating with real robot)
        self.robot.motors(motor_values)

        # Sleep for a short duration to simulate control loop timing
        # exit(0)
        time.sleep(0.05)  # Adjust the sleep time as needed

        # Stop the robot
        # motor_values_scaledown = np.array(motor_values)
        # motor_values_scaledown_int = []
        # for mv in motor_values_scaledown:
        #    motor_values_scaledown_int.append(int)
        # motor_values = motor_values_scaledown_int
        # self.robot.motors(motor_values)
        self.stop_drive()

    def send_to_XY_Theta_simple(self, target_x, target_y, orientation_theta, left_motor=0, right_motor=9):
        disable_Kdist = False
        if disable_Kdist:
          Kp_dist = 0.0
          Ki_dist = 0.0
          Kd_dist = 0.000  # 0.1
        else:
          Kp_dist = 6
          Ki_dist = 0.01
          Kd_dist = 0.00  # 0.1
        Kp_theta = 10
        Ki_theta = 0.01
        Kd_theta = 0.000  # 0.1

        # Compute distance and angle to target
        dx = target_x - self.x
        dy = target_y - self.y
        distance_error = np.sqrt(dx**2 + dy**2)
        target_angle = np.degrees(np.arctan2(dy, dx))
        angle_error = target_angle - self.angle - orientation_theta
        angle_error = (angle_error + 180) % 360 - 180  # Normalize to [-180, 180]

        print(f'dx = {dx}, dy = {dy}')
        print(f'distance_error = {distance_error}, target_angle = {target_angle}, angle_error = {angle_error}')

        # PID calculations for distance
        self.dist_integral += distance_error
        dist_derivative = distance_error - self.prev_dist_error
        self.prev_dist_error = distance_error

        P_dist = Kp_dist * distance_error
        I_dist = Ki_dist * self.dist_integral
        D_dist = Kd_dist * dist_derivative

        print(f'P_dist = {P_dist}, I_dist = {I_dist}, D_dist = {D_dist}')
        print(f'dist_integral = {self.dist_integral}, dist_derivative = {dist_derivative}')

        # PID calculations for angle
        self.theta_integral += angle_error
        theta_derivative = angle_error - self.prev_theta_error
        self.prev_theta_error = angle_error

        P_theta = Kp_theta * angle_error
        I_theta = Ki_theta * self.theta_integral
        D_theta = Kd_theta * theta_derivative

        print(f'P_theta = {P_theta}, I_theta = {I_theta}, D_theta = {D_theta}')
        print(f'theta_integral = {self.theta_integral}, theta_derivative = {theta_derivative}')

        # Compute motor outputs
        left_motor_speed = P_dist + I_dist + D_dist + P_theta + I_theta + D_theta
        right_motor_speed = P_dist + I_dist + D_dist - (P_theta + I_theta + D_theta)

        print(f'left_motor_speed (before clip) = {left_motor_speed}, right_motor_speed (before clip) = {right_motor_speed}')

        # Clip motor speeds to acceptable range
        left_motor_speed = np.clip(left_motor_speed, -127, 127)
        right_motor_speed = -np.clip(right_motor_speed, -127, 127)

        print(f'left_motor_speed (after clip) = {left_motor_speed}, right_motor_speed (after clip) = {right_motor_speed}')

        # Set motor speeds
        motor_values = self.robot.motor
        motor_values[left_motor] = int(left_motor_speed)
        motor_values[right_motor] = int(right_motor_speed)

        print(f'motor_values[{left_motor}] = {motor_values[left_motor]}, motor_values[{right_motor}] = {motor_values[right_motor]}')

        # Simulate sending motor values to robot (uncomment when integrating with real robot)
        self.robot.motors(motor_values)

        # Sleep for a short duration to simulate control loop timing
        # exit(0)
        time.sleep(0.05)  # Adjust the sleep time as needed

        # Stop the robot
        # motor_values_scaledown = np.array(motor_values)
        # motor_values_scaledown_int = []
        # for mv in motor_values_scaledown:
        #    motor_values_scaledown_int.append(int)
        # motor_values = motor_values_scaledown_int
        # self.robot.motors(motor_values)
        self.stop_drive()

    def drive(self, args):
        direction, speed, drive_time = args
        left_motor=0
        right_motor=9
        if direction == "forward":
          left_drive = 1
          right_drive = -1
        elif direction == "backward":
          left_drive = -1
          right_drive = 1
        else:
          left_drive = 0
          right_drive = 0
        motor_values = self.robot.motor
        # for i in range(max(int(drive_time * 10),1)):
        motor_values[left_motor] = left_drive * speed
        motor_values[right_motor] = right_drive * speed
        self.robot.motors(motor_values)
          # time.sleep(1/10)
        time.sleep(drive_time)
        self.stop_drive()
    
    def stop_drive(self):
        motor_values = 10 * [0]
        self.robot.motors(motor_values)
    def getSensorValues(self):
        return self.robot.sensors()
    def claw(self, args):
        # value, action=clawAction.Close, claw_motor=1, drive_time=0.5
        value, action, claw_motor, drive_time = args
        motor_values = self.robot.motor
        if action == clawAction.Close:
            motor_values[claw_motor] = -1 * value
        else:
            motor_values[claw_motor] = 1 * value
        self.robot.motors(motor_values)
        time.sleep(drive_time)
        self.stop_drive()
        if self.robot.sensors()[2] == 1 and action == clawAction.Close:
            print("ball held")
            return 1
        else:
            print("ball missed")
            return 0

    def update_robot_move_arm(self, args):
      # armPosition=ARM_POSITION.low, motor=2, error=20
      print("updating robot move arm", args)
      armPosition, motor, error = args
      POTENTIOMETRIC_SENSOR_MAX_VALUE = 2582
      POTENTIOMETRIC_SENSOR_MIN_VALUE = 1587
      goal = POTENTIOMETRIC_SENSOR_MIN_VALUE
      if armPosition == "low":
        goal = POTENTIOMETRIC_SENSOR_MIN_VALUE
      elif armPosition == "mid":
        goal = (POTENTIOMETRIC_SENSOR_MAX_VALUE + POTENTIOMETRIC_SENSOR_MIN_VALUE) / 2
      elif armPosition == "high":
        goal = POTENTIOMETRIC_SENSOR_MAX_VALUE
      # don't take less then 5 seconds to move the arm
      start_time = time.time()
      while True:
        try:
          sensor_values = self.robot.sensors()
          if len(sensor_values) == 0:
            print("no sensor values")
            continue
        except:
          print("error reading sensor values")
          continue
        if sensor_values[0] > goal:
          self.robot.motor[motor] = -15
        else:
          self.robot.motor[motor] = 60
        if np.abs(sensor_values[0] - goal) < error:
          break
        # time.sleep(0.1)
      # self.robot.motor[motor] =  30
      if armPosition == 'high' or armPosition == 'mid':
        self.robot.motor[motor] =  10
        time.sleep(0.5)
      
    def update_robot_gotoV1(self, goal):
      dpos = [goal[0], goal[1]] 
      dist = np.sqrt(dpos[0] ** 2 + dpos[1] ** 2)
      theta = np.degrees(np.arctan2(dpos[1], dpos[0]))
      theta = (theta + 180) % 360 - 180  # [-180, 180]
      Pforward = 30
      Ptheta = 1
      if np.abs(theta) < 30:
          self.robot.motor[0] = int(-Pforward * dist + Ptheta * theta)
          self.robot.motor[9] = int(Pforward * dist + Ptheta * theta)
      else:
          self.robot.motor[0] = int(127 if theta > 0 else -127)
          self.robot.motor[9] = int(127 if theta > 0 else -127)
      # if dist < 1 and np.abs(theta) > 30:
      #     self.robot.motor[0] = int(127 if theta > 0 else -127)
      #     self.robot.motor[9] = int(127 if theta > 0 else -127)
      time.sleep(0.2)
    
    def update_robot_gotoV2(self, goal, offset=90):
      dpos = [goal[0], goal[1]] 
      dist = np.sqrt(dpos[0] ** 2 + dpos[1] ** 2)
      theta = np.degrees(np.arctan2(dpos[0], dpos[1])) #+ offset
      theta = (theta + 180) % 360 - 180  # [-180, 180]
      print(f'dist = {dist}, theta = {theta}')
      Pforward = 50
      Ptheta = 5
      motor_values = self.robot.motor
      motor_values[0] = int(Pforward * dist + Ptheta * theta)
      motor_values[9] = int(-Pforward * dist + Ptheta * theta)
      self.robot.motors(motor_values)
      time.sleep(0.5)

    def update_robot_goto(self, goal,left_motor=0, right_motor=9):
        dpos = [goal[0], goal[1]] 
        dist = np.sqrt(dpos[0] ** 2 + dpos[1] ** 2)
        theta = np.degrees(np.arctan2(dpos[0], dpos[1]))
        # thetaWithNormal = 90 - theta
        # theta = thetaWithNormal
        theta = (theta + 180) % 360 - 180  # [-180, 180]
        print(f'dist = {dist}, theta = {theta}')
        # exit(0)
        # PI Control for distance
        if dist < 10:
           Kp_dist = 5
        else:
            Kp_dist = 1
        # Kp_dist = 5
        Ki_dist = 0.000
        dist_error = dist
        self.dist_integral += dist_error
        P_dist = Kp_dist * dist_error
        I_dist = Ki_dist * self.dist_integral

        # PI Control for orientation
        if np.abs(theta) < 30:
            Kp_theta = -1
        else:
            Kp_theta = -0.5
        # Kp_theta = 2
        Ki_theta = 0.000
        theta_error = theta
        self.theta_integral += theta_error
        P_theta = Kp_theta * theta_error
        I_theta = Ki_theta * self.theta_integral
        # MAX_VALUE_FOR_ROTATION = 60
        motor_values = self.robot.motor
        if ( np.abs(theta) > 1 or dpos[1] > 1):
          motor_values[left_motor] = int(P_dist * dist + P_theta * theta + I_dist + I_theta)
          motor_values[right_motor] = int(-P_dist * dist + P_theta * theta + I_dist + I_theta)
        # else:
        #     motor_values[left_motor] = int(-MAX_VALUE_FOR_ROTATION if theta > 0 else MAX_VALUE_FOR_ROTATION)
        #     motor_values[right_motor] = int(-MAX_VALUE_FOR_ROTATION if theta > 0 else MAX_VALUE_FOR_ROTATION)
        self.robot.motors(motor_values)
        # if dist < 1 and np.abs(theta) > 30:
        #     motor_values[left_motor] = int(MAX_VALUE_FOR_ROTATION if theta > 0 else -MAX_VALUE_FOR_ROTATION)
        #     motor_values[right_motor] = int(MAX_VALUE_FOR_ROTATION if theta > 0 else -MAX_VALUE_FOR_ROTATION)
        # Update previous errors
        self.prev_dist_error = dist_error
        self.prev_theta_error = theta_error
        effective_sleep_time = 0.2 # make this a function of theta, and distance if needed
        #print all debug values here
        print(f'x = {goal[0]}, y = {goal[1]}')
        print(f'dist = {dist}, theta = {theta}')
        print(f'P_dist = {P_dist}, P_theta = {P_theta}, I_dist = {I_dist}, I_theta = {I_theta}')
        print(f'motor[0] = {motor_values[0]}, motor[9] = {motor_values[9]}')
        time.sleep(effective_sleep_time)

#     ROTATION_DIRECTION = {
#     "counter_clockwise": 1,
#     "clockwise": -1
# }
 # revisit this code after IMU integration
 #https://chat.openai.com/share/85c1d999-2fc5-4ad0-916e-f22d141afb2d
    def rotateRobotPI(self, theta, speed=MINIMUM_INPLACE_ROTATION_SPEED):
        tolerance = 1
        current_angle = 90 # replace this with imu reading
        error = theta - current_angle
        if error > 0:
           self.setSearchDirection("clockwise")
        else:
            self.setSearchDirection("counter_clockwise")
        error = (error + 180) % 360 - 180  # [-180, 180]
        if abs(error) < tolerance:
          return 
        Kp = 10
        Ki = 0
        self.theta_integral += error
        pi_output = Kp * error + Ki * self.theta_integral
        # Map the PI output to the motor speeds
        motor_values = self.robot.motor
        motor_values[0] = np.clip(int(speed * np.sign(pi_output)), -speed, speed)
        motor_values[9] = np.clip(int(speed * np.sign(pi_output)), -speed, speed)
        # self.robot.motor[0] = np.clip(int(speed * np.sign(pi_output)), -speed, speed)
        # self.robot.motor[9] = np.clip(int(speed * np.sign(pi_output)), -speed, speed)
        print(f'error = {error}, pi_output = {pi_output}')
        # self.robot.motors(motor_values)
        print(f'motor[0] = {motor_values[0]}, motor[9] = {motor_values[9]}')
        time.sleep(0.2)
        self.stop_drive()

    def rotateRobot(self, args):#seconds, dir=-1, speed=MINIMUM_INPLACE_ROTATION_SPEED):
        seconds, dir, speed = args
        # for i in range(int(seconds*100)):
        self.robot.motor[0] = speed * dir
        self.robot.motor[9] = speed * dir
            # time.sleep(1/100)
            # self.stop_drive()
        time.sleep(seconds)
        self.stop_drive()

    def testRotate(self, rot_speed=30, rot_time=1, rot_dir=1):
        self.rotateRobot(rot_time, rot_dir, rot_speed)
        time.sleep(0.5)
        self.rotateRobot(rot_time, -rot_dir, rot_speed)
        time.sleep(0.5)
        self.stop_drive()
        
    
    def testAngle(self):
        goal_angle = np.random.rand() * 360 - 180
        self.update_robot_goto(self.x, self.y, goal_angle)
        self.update_robot_goto(self.x, self.y, -goal_angle)
    
    def testTransulateAlongY(self):
            if self.robot.running():
                if self.Y > self.set_Y:
                    self.drive_forward(30)
                    self.stop_drive()
                else:
                    self.drive_backward(30)
                    self.stop_drive()
    def calibrateMotors(self):
        motor_values = self.robot.motor
        motor_values[0] = -127
        motor_values[9] = -127
        self.robot.motors(motor_values)
        time.sleep(0.2)
        # time.sleep(1)
        # self.stop_drive()
        # time.sleep(0.5)
        # motor_values[0] = -30
        # motor_values[9] = -30
        # self.robot.motors(motor_values)
        # time.sleep(1)
        # self.stop_drive()
        # time.sleep(0.5)
        # motor_values[0] = 0
        # motor_values[9] = 0
        # self.robot.motors(motor_values)
        # time.sleep(0.5)
        # self.stop_drive()
        # time.sleep(0.5)
        # motor_values[1] = 30
        # self.robot.motors(motor_values)
        # time.sleep(1)
        # self.stop_drive()
        # time.sleep(0.5)
        # motor_values[1] = -30
        # self.robot.motors(motor_values)
        # time.sleep(1)
        # self.stop_drive()
        # time.sleep(0.5)
        # motor_values[1] = 0
        # self.robot.motors(motor_values)
        # time.sleep(0.5)
        # self.stop_drive()
        # time.sleep(0.5)
        # motor_values[2] = 30
        # self.robot.motors(motor_values)
        # time.sleep(1)
        # self.stop_drive()
        # time.sleep(0.5)
        # motor_values[2] = -30
        # self.robot.motors(motor_values)
        # time.sleep(1)
        # self.stop_drive()
        # time.sleep(0.5)
        # motor_values[2] = 0
        # self.robot.motors(motor_values)
        # time.sleep(0.5)
        # self.stop

def listener(queue, vex_control):
    while True:
        message = queue.get()
        if message:
            command, args = message
            if hasattr(vex_control, command):
                method = getattr(vex_control, command)
                method(args)

import select
import curses
import time      
import cv2   

def keyboard_control(stdscr, control, robot):
    # Set up curses environment
    curses.cbreak()
    stdscr.nodelay(True)
    stdscr.clear()

    while True:
        char = stdscr.getch()

        if char == ord('w'):
            # control.drive(direction="forward", speed=30, drive_time=1)
            control.drive(['forward', 30, 0.7])
        elif char == ord('s'):
            # control.drive(direction="backward", speed=30, drive_time=1)
            control.drive(['backward', 30, 0.7])
        elif char == ord('a'):
            control.rotateRobot([0.05, ROTATION_DIRECTION["counter_clockwise"], MINIMUM_INPLACE_ROTATION_SPEED])
        elif char == ord('d'):
            control.rotateRobot([0.05, ROTATION_DIRECTION["clockwise"], MINIMUM_INPLACE_ROTATION_SPEED]) 
        elif char == ord('['):
            control.claw([20, "close", 1, 0.8])
        elif char == ord(']'):
            control.claw([20, "open", 1, 1.5])
        elif char == ord('-'):
            control.update_robot_move_arm([ARM_POSITION.low, 2, 20])
            control.stop_drive()
        elif char == ord('='):  # Note: '+' is usually combined with shift, so '=' is used
            control.update_robot_move_arm([ARM_POSITION.high, 2, 20])
        elif char == ord('q'):
            control.stop_drive()
            robot.stop()
            exit(0)


        time.sleep(0.1)
        # control.stop_drive()

        if char == curses.KEY_RESIZE:
            stdscr.clear()  # Clear the screen if terminal size changes

def execute_and_respond(client_socket, control, command, args):
    # This function will execute the command and send the response back to the client
    try:
        print(f"Executing command: {command} with args: {args}")
        if callable(getattr(control, command)):
            method = getattr(control, command)
            result = method(args) if args else method()  # Execute the method
            time.sleep(3)
            result = [[control.robot.sensors()[0], control.robot.sensors()[1], control.robot.sensors()[1]], result]
            response = json.dumps({'status': 'success', 'result': result})
        else:
            response = json.dumps({'status': 'error', 'message': 'Command not callable'})
    except Exception as e:
        response = json.dumps({'status': 'error', 'message': str(e)})

    client_socket.send(response.encode())
    client_socket.close()

def keyboard():
    robot = VexCortex("/dev/ttyUSB0")
    control = VexControl(robot)
    control.stop_drive()
    # exit(0)
    manual_control = True
    # manual_control = False
    if manual_control:
      curses.wrapper(keyboard_control, control, robot)
    #
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', 6000))
    server_socket.listen(1)
    server_socket.setblocking(0)  # Set the socket to non-blocking mode

    # input argument is stop
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        control.stop_drive()
        robot.stop()
        exit(0)
    import signal
    def signal_handler(sig, frame):
        control.stop_drive()
        robot.stop()
        exit(0)
    # Also stop the robot if the program is terminated or user presses Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    latest_command = None
    error_margin = [5, 0,0]
    while True:
      previous_sensor_values = robot.sensors_external()
      if len(previous_sensor_values) > 0:
        print("sensor values = ", previous_sensor_values)
        break

    while True:
        time.sleep(0.1)
        current_sensor_values = robot.sensors_external()
        if len(current_sensor_values) > 0 :
          all_errors_within_limit = all(abs(a - b) <= c for a, b, c in zip(previous_sensor_values, current_sensor_values, error_margin))
          if not all_errors_within_limit:
              print("sensor values = ", current_sensor_values)
              previous_sensor_values = current_sensor_values
        
        ready = select.select([server_socket], [], [], 0.1)  # Non-blocking select
        if ready[0]:
            client_socket, addr = server_socket.accept()
            message = client_socket.recv(1024)
            # client_socket.close()
            # Update the latest command
            latest_command = json.loads(message)
        if latest_command:
            command = latest_command['command']
            args = latest_command.get('args', [])
            print(f"Executing command: {command} with args: {args}")
            execute_and_respond(client_socket, control, command, args)
            latest_command = None  # Reset after execution
        # Check and execute only the latest command



    #untested code
    # control.update_robot_goto([-15,38])
    # move the robot arm 
    # print sensor values for 3 seconds
    # while True:
    #   print(robot.sensors())
    #   time.sleep(0.1)
    # print(robot.sensors())
    # start_time = time.time()
    # while time.time() - start_time < 3:
    # control.update_robot_move_arm(armPosition=ARM_POSITION.low)
    #   print(robot.sensors())
    # print("done")
    # time.sleep(3)
        # Also stop the robot if the program is terminated or user presses Ctrl+C
    # signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGTERM, signal_handler)
    control.stop_drive()
    robot.stop()

import time
import logging 
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class RobotControl:
    def __init__(self, robot):
        self.robot = robot
        self.left_motor = 0
        self.right_motor = 9
        self.arm_motor = 2
        self.claw_motor = 1
        self.ball_held = False

    def rotate_robot(self, value):
        max_speed = 80
        min_speed = 50
        measurement_range = 128
        multi_factor = (max_speed-min_speed)/measurement_range
        if value > 160:
            direction = -1  # counterclockwise
            logging.debug(f"Rotating counter clkwise")
        elif value < 100:
            direction = 1  # clockwise 
            logging.debug(f"Rotating clkwise")
        else:
           direction = 0
          #  speed = 0
        
        speed = min(min_speed + abs(measurement_range - value)*multi_factor, max_speed)

        speed = int(speed)
        left_motor_speed = direction * speed
        right_motor_speed = direction * speed

        motor_values = self.robot.motor
        motor_values[self.left_motor] = int(left_motor_speed)
        motor_values[self.right_motor] = int(right_motor_speed)

        self.robot.motors(motor_values)

    def drive(self, value):
        max_speed = 80
        min_speed = 50
        measurement_range = 128
        multi_factor = (max_speed-min_speed)/measurement_range
        # pressing down = 255, up = 0
        if value > 160:
            direction = -1  # backward
            logging.debug(f"Driving backward")
        elif value < 100:
            direction = 1  # forward
            logging.debug(f"Driving forward")
        else:
           direction = 0
           speed = 0

        speed = min(min_speed + abs(measurement_range - value)*multi_factor, max_speed)
        speed = int(speed)
        left_drive = direction 
        right_drive = -1 * direction
        left_motor_speed = left_drive * speed
        right_motor_speed = right_drive * speed

        motor_values = self.robot.motor
        motor_values[self.left_motor] = int(left_motor_speed)
        motor_values[self.right_motor] = int(right_motor_speed)

        self.robot.motors(motor_values)
        # time.sleep(0.05)

    def move_claw(self, value):
        if value > 140:
            speed = 50
            direction = 1  # open
            logging.debug(f"opening claw: {speed}")
        elif value < 116:
            speed = 50
            direction = -1  # close
            logging.debug(f"Closing claw: {speed}")
        else:
           direction = 0
           speed = 0

        motor_values = self.robot.motor
        motor_values[self.claw_motor] = int(direction * speed)
        self.robot.motors(motor_values)

    def move_arm(self, value):
        if value < 116: #> 140:
            speed = 50
            direction = 1  # up
            logging.debug(f"Moving arm up: {speed}")
            self.ball_held = True
        elif value > 140 :#< 116:
            speed = 30
            direction = -1  # down
            self.ball_held = False
            logging.debug(f"Moving arm down: {speed}")
        else:
           if self.ball_held: #continue keeping it up
              speed = 30
              direction = 1  # up
           else:
               direction = -1 
               speed = 0

        motor_values = self.robot.motor
        motor_values[self.arm_motor] = int(direction * speed)
        self.robot.motors(motor_values)

    def stop_drive(self):
        logging.debug(f"Stopping the robot")
        motor_values = self.robot.motor
        motor_values[self.left_motor] = 0
        motor_values[self.right_motor] = 0
        motor_values[self.arm_motor] = 0
        motor_values[self.claw_motor] = 0
        self.robot.motors(motor_values)


import evdev
from evdev import InputDevice, categorize, ecodes
import time
import numpy as np

# Find the controller device
devices = [evdev.InputDevice(fn) for fn in evdev.list_devices()]
controller = None
for device in devices:
    if 'Wireless Controller' in device.name:  # Adjust this string if necessary to match your controller's name
        controller = device
        break
    else:
        print(device.name)

if controller is None:
    print("No Sony controller found.")
    exit()

print(f"Using device: {controller.name} ({controller.fn})")


def wireless_controller():
    robot = VexCortex("/dev/ttyUSB0")
    # control = VexControl(robot)
    # control.stop_drive()

    control = RobotControl(robot)
    control.stop_drive()

    for event in controller.read_loop():
        if event.type == ecodes.EV_KEY:
            key_event = categorize(event)
            if key_event.keystate == key_event.key_down or key_event.keystate == key_event.key_up:
                if key_event.keycode in ['BTN_TR2', 'BTN_TL2']:
                    control.stop_drive()
        elif event.type == ecodes.EV_ABS:
            abs_event = categorize(event)
            if abs_event.event.code == 0:  # Left joystick sideways
                control.rotate_robot(abs_event.event.value)
            elif abs_event.event.code == 1:  # Left joystick up-down
                control.drive(abs_event.event.value)
            elif abs_event.event.code == 3:  # Right joystick sideways
                control.move_claw(abs_event.event.value)
            elif abs_event.event.code == 4:  # Right joystick up-down
                control.move_arm(abs_event.event.value)

        # input argument is stop
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        control.stop_drive()
        robot.stop()
        exit(0)
    import signal
    def signal_handler(sig, frame):
        control.stop_drive()
        robot.stop()
        exit(0)
    # Also stop the robot if the program is terminated or user presses Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
  # keyboard()
  wireless_controller()