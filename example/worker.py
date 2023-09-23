from vex_serial import VexCortex
import camera
import time

from enum import Enum
class clawAction(Enum):
  Open = 1
  Close = -1

robot = VexCortex("/dev/ttyUSB0")

def drive_forward(robot, value, drive_time=1, left_motor=0, right_motor=9):
  motor_values = robot.motor
  left_drive = 1
  right_drive = -1
  motor_values[left_motor] = left_drive * value
  motor_values[right_motor] = right_drive * value
  robot.motors(motor_values)
  time.sleep(drive_time)
  stop_drive(robot)

def drive_backward(robot, value, drive_time=1, left_motor=0, right_motor=9):
  motor_values = robot.motor
  left_drive = -1
  right_drive = 1
  motor_values[left_motor] = left_drive * value
  motor_values[right_motor] = right_drive * value
  time.sleep(drive_time)
  stop_drive(robot)

def stop_drive(robot):
  motor_values = 10*[0]
  robot.motors(motor_values)



def claw(robot, value, action = clawAction.Close, claw_motor=1):
  motor_values = robot.motor
  if action == clawAction.close:
    motor_values[claw_motor] = -1 * value
  else:
    motor_values[claw_motor] = -1 * value
  robot.motors(motor_values)
  stop_drive()

if robot.running():
  drive_backward(robot,60)
  stop_drive(robot)

