import sys
import glob
import serial
import time
from multiprocessing import Process, Queue, Array, RawValue, Value
from ctypes import c_double, c_bool, c_int
import socket
import json
import copy
import logging
import evdev
from evdev import InputDevice, categorize, ecodes
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

ROTATION_DIRECTION = {
    "counter_clockwise": 1,
    "clockwise": -1
}

MINIMUM_INPLACE_ROTATION_SPEED = 60

class ARM_POSITION(Enum):
    low = 'low'
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

        time.sleep(0.005)  # throttle to prevent CPU overload

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

    def _autofind_path(self):
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
                break
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
        return self._motor_values

    def motors(self, motor_values):
        self._motor_values.set(motor_values)

    @property
    def sensor(self):
        return self._sensor_values

    def sensors(self):
        self._sensor_values._data.acquire()
        num_sensors = self._num_sensors.value
        sensor_values = self._sensor_values._data[:num_sensors]
        self._sensor_values._data.release()
        return sensor_values

    def sensors_external(self):
        self._sensor_values._data.acquire()
        num_sensors = self._num_sensors.value
        sensor_values = self._sensor_values._data[:num_sensors]
        self._sensor_values._data.release()
        return_sensor_values = []
        for sensor_value in sensor_values:
            return_sensor_values.append(sensor_value)
        return return_sensor_values

class RobotControl:
    def __init__(self, robot):
        self.robot = robot
        self.left_motor = 0
        self.right_motor = 9
        self.arm_motor = 2
        self.claw_motor = 1

    def rotate_robot(self, value):
        if value > 160:
            speed = min(30 + abs(value - 128), 127)
            direction = 1  # clockwise
            logging.debug(f"Rotating clockwise with value: {speed}")
        elif value < 100:
            speed = min(30 + abs(128 - value), 127)
            direction = -1  # counterclockwise
            logging.debug(f"Rotating counterclockwise with value: {speed}")
        else:
            direction = 0
            speed = 0

        left_motor_speed = direction * speed
        right_motor_speed = -direction * speed

        motor_values = self.robot.motor
        motor_values[self.left_motor] = int(left_motor_speed)
        motor_values[self.right_motor] = int(right_motor_speed)

        self.robot.motors(motor_values)

    def drive(self, value):
        if value > 160:
            speed = min((30 + abs(128 - value)), 60)
            direction = -1  # backward
            logging.debug(f"Driving backward with value: {speed}")
        elif value < 100:
            speed = min((30 + abs(128 - value)), 60)
            direction = 1  # forward
            logging.debug(f"Driving forward with value: {speed}")
        else:
            direction = 0
            speed = 0

        left_drive = direction
        right_drive = -1 * direction
        left_motor_speed = left_drive * speed
        right_motor_speed = right_drive * speed

        motor_values = self.robot.motor
        motor_values[self.left_motor] = int(left_motor_speed)
        motor_values[self.right_motor] = int(right_motor_speed)

        self.robot.motors(motor_values)

    def move_claw(self, value):
        if value > 140:
            speed = 50
            direction = 1  # open
            logging.debug(f"Opening claw: {speed}")
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
        if value > 140:
            speed = 80
            direction = 1  # up
            logging.debug(f"Moving arm up: {speed}")
        elif value < 116:
            speed = 40
            direction = -1  # down
            logging.debug(f"Moving arm down: {speed}")
        else:
            direction = 0
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

def wireless_controller():
    robot = VexCortex("/dev/ttyUSB0")
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
        
        # Short delay to prevent CPU overload
        time.sleep(0.01)

        # Input argument is stop
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
    wireless_controller()
