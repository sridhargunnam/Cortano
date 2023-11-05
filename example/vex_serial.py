import sys
import glob
import serial
import time
from multiprocessing import Process, Array, RawValue, Value
from ctypes import c_double, c_bool, c_int

ROTATION_DIRECTION = {
    "counter_clockwise": 1,
    "clockwise": -1
}

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


import time
import numpy as np

class clawAction:
    Close = "close"
    Open = "open"
    Stop = "stop"


class VexControl:
    def __init__(self, robot):
        self.robot = robot
    
    def drive(self, direction, speed, drive_time=1, left_motor=0, right_motor=9):
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
        motor_values[left_motor] = left_drive * speed
        motor_values[right_motor] = right_drive * speed
        self.robot.motors(motor_values)
        time.sleep(drive_time)
        self.stop_drive()
    
    def stop_drive(self):
        motor_values = 10 * [0]
        self.robot.motors(motor_values)
    
    def claw(self, value, action=clawAction.Close, claw_motor=1, drive_time=0.5):
        motor_values = self.robot.motor
        if action == clawAction.Close:
            motor_values[claw_motor] = -1 * value
        else:
            motor_values[claw_motor] = 1 * value
        self.robot.motors(motor_values)
        time.sleep(drive_time)
        self.stop_drive()

    def update_robot_move_arm(self, armPosition=ARM_POSITION.low, motor=2, error=20):
      POTENTIOMETRIC_SENSOR_MAX_VALUE = 2582
      POTENTIOMETRIC_SENSOR_MIN_VALUE = 1587
      if armPosition == ARM_POSITION.low:
        goal = POTENTIOMETRIC_SENSOR_MIN_VALUE
      elif armPosition == ARM_POSITION.mid:
        goal = (POTENTIOMETRIC_SENSOR_MAX_VALUE + POTENTIOMETRIC_SENSOR_MIN_VALUE) / 2
      elif armPosition == ARM_POSITION.high:
        goal = POTENTIOMETRIC_SENSOR_MAX_VALUE
      # don't take less then 5 seconds to move the arm
      start_time = time.time()
      while time.time() - start_time < 5:
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
          self.robot.motor[motor] = 35
        if np.abs(sensor_values[0] - goal) < error:
          break
        time.sleep(0.1)
      # self.robot.motor[motor] =  30
      if armPosition == 'high' or armPosition == 'mid':
        self.robot.motor[motor] =  30
      

    def update_robot_goto(self, state, goal):
        dpos = [goal[0] - state[0] , goal[1] - state[1]] 
        dist = np.sqrt(dpos[0] ** 2 + dpos[1] ** 2)
        theta = np.degrees(np.arctan2(dpos[1], dpos[0]))
        theta = (theta + 180) % 360 - 180  # [-180, 180]
        Pforward = 30
        Ptheta = 30
        if np.abs(theta) < 30:
            self.robot.motor[0] = -Pforward * dist + Ptheta * theta
            self.robot.motor[9] = Pforward * dist + Ptheta * theta
        else:
            self.robot.motor[0] = 127 if theta > 0 else -127
            self.robot.motor[9] = 127 if theta > 0 else -127
        if dist < 1 and np.abs(theta) > 30:
            self.robot.motor[0] = 127 if theta > 0 else -127
            self.robot.motor[9] = 127 if theta > 0 else -127
    
    def rotateRobot(self, seconds, dir, speed):
        self.robot.motor[0] = 35 * dir
        self.robot.motor[9] = 35 * dir
        time.sleep(0.1)
        for i in range(seconds*1000):
            self.robot.motor[0] = speed * dir
            self.robot.motor[9] = speed * dir
            time.sleep(1/1000)
            self.stop_drive()
        # self.robot.motor[0] = speed * dir
        # self.robot.motor[9] = speed * dir
        # time.sleep(seconds)
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

MINIMUM_INPLACE_ROTATION_SPEED = 35

if __name__ == "__main__":
    robot = VexCortex("/dev/ttyUSB0")
    control = VexControl(robot)

    #tested code
    # control.drive(direction="forward", speed=30, drive_time=7)
    # control.drive(direction="backward", speed=30, drive_time=2)
    # control.rotateRobot(seconds=5, dir=ROTATION_DIRECTION["counter_clockwise"], speed=MINIMUM_INPLACE_ROTATION_SPEED)
    # control.rotateRobot(seconds=1, dir=ROTATION_DIRECTION["clockwise"], speed=MINIMUM_INPLACE_ROTATION_SPEED)
    # control.claw(20, clawAction.Open, drive_time=1.5)
    # control.claw(20, clawAction.Close, drive_time=1.5)
    '''    control.claw(20, clawAction.Close, drive_time=1.5)
    start_time = time.time()
    while time.time() - start_time < 5:
        print(robot.sensors())
    # [1587, 0, 1] the last sensor corresponds to the bottom limit switch, and will be 1 if the claw is closed
    
    control.claw(30, clawAction.Open, drive_time=0.8)
    start_time = time.time()
    while time.time() - start_time < 5:
        print(robot.sensors())'''
    
    #untested code
    # move the robot arm 
    # print sensor values for 3 seconds
    start_time = time.time()
    while time.time() - start_time < 3:
      control.update_robot_move_arm(armPosition=ARM_POSITION.mid)
      print(robot.sensors())
    print("done")
    time.sleep(3)
    control.stop_drive()
    robot.stop()
