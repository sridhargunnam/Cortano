from interface import SimInterface
import numpy as np

def update_robot_goto(robot, state, goal):
  dpos = np.array(goal) - state[:2]
  dist = np.sqrt(dpos[0] ** 2 + dpos[1] ** 2)
  theta = np.degrees(np.arctan2(dpos[1], dpos[0])) - state[2]
  theta = (theta + 180) % 360 - 180 # [-180, 180]
  Pforward = 30
  Ptheta = 8

  # restrict operating range
  if np.abs(theta) < 30:
  #   # P-controller
    robot.motor[0] = -Pforward * dist + Ptheta * theta
    robot.motor[9] =  Pforward * dist + Ptheta * theta
  else:
    robot.motor[0] = 127 if theta > 0 else -127
    robot.motor[9] = 127 if theta > 0 else -127

def update_robot_move_arm(robot, angle, goal):
  # P-controller with constant current
  robot.motor[1] = (goal - angle) * 127 + 30

if __name__ == "__main__":
  # robot = RemoteInterface("...")
  robot = SimInterface()

  while True:
    robot.update()
    # color, depth, sensors = robot.read()
    # x, y, theta = RGBDOdometry()
    sensors = robot.read()
    print(sensors)
    x, y = robot.pos
    theta = robot.angle # get these from SLAM

    # positive values make the robot turn left
    # negative value on 0, positive value on 9 make the robot go forward
    goal = (10, 0)
    # goal = objdetection from PyTorch
    state = (x, y, theta)
    # update_robot_goto(robot, state, goal)

    upper_limit = 2625
    lower_limit = 604
    upper_degree = 30
    lower_degree = -32
    arm_angle = (sensors[0] - lower_limit) / (upper_limit - lower_limit) * (upper_degree - lower_degree) + lower_degree

    update_robot_move_arm(robot, arm_angle, 29)