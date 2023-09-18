from interface import SimInterface
import numpy as np


def update_robot_goto(robot, state, goal):
  dpos = np.array(goal) - state[:2]
  dist = np.sqrt(dpos[0] ** 2 + dpos[1] ** 2)
  theta = np.degrees(np.arctan2(dpos[1], dpos[0])) - state[2]
  theta = (theta + 180) % 360 - 180 # [-180, 180]
  Pforward = 30
  Ptheta = 30
  # restrict operating range
  if np.abs(theta) < 30:
  #   # P-controller
    robot.motor[0] = -Pforward * dist + Ptheta * theta
    robot.motor[9] =  Pforward * dist + Ptheta * theta
  else:
    # turn in place
    robot.motor[0] = 127 if theta > 0 else -127
    robot.motor[9] = 127 if theta > 0 else -127

  # if the robot is close to the goal position, but if there is a large angle difference
  # then the robot should turn in place
  # TODO: tune the parameters to fix the bug. The bug is that the robot will not turn in place when the position is close to the goal. 
  if dist < 1 and np.abs(theta) > 30:
    robot.motor[0] = 127 if theta > 0 else -127
    robot.motor[9] = 127 if theta > 0 else -127

def update_robot_move_arm(robot, angle, goal):
  # P-controller with constant current
  # +30 is to keep the arm from falling
  robot.motor[1] = (goal - angle) * 127 + 30

if __name__ == "__main__":
  # robot = RemoteInterface("...")
  robot = SimInterface()
  
  # goal = (10, 0)
  #generate a random goal
  for i in range(5):
    goal_pos = np.random.rand(2) * 50
    # goal angle between -180 and 180
    goal_angle = np.random.rand() * 360 - 180
    while True:
      # print current state and goal, well formatted
      print("Current state: ", robot.pos, robot.angle)
      print("Goal    State: ", goal_pos)
      ErrorPos = np.sqrt((robot.pos[0] - goal_pos[0]) ** 2 + (robot.pos[1] - goal_pos[1]) ** 2)
      ErrorAngle = np.degrees(np.arctan2(goal_pos[1] - robot.pos[1], goal_pos[0] - robot.pos[0])) - robot.angle
      print("Error   Pos: ", ErrorPos)
      print("Error   Angle: ", ErrorAngle)
      # if state is close to goal
      dist = np.sqrt((robot.pos[0] - goal_pos[0]) ** 2 + (robot.pos[1] - goal_pos[1]) ** 2)
      delta_angle = abs(np.degrees(np.arctan2(goal_pos[1] - robot.pos[1], goal_pos[0] - robot.pos[0])) - robot.angle)
      if dist < 1 and delta_angle < 1:
        break
      robot.set_goal(goal_pos, goal_angle)
      robot.update()
      # color, depth, sensors = robot.read()
      # x, y, theta = RGBDOdometry()
      sensors = robot.read()
      # print(sensors)
      x, y = robot.pos
      theta = robot.angle # get these from SLAM
      # positive values make the robot turn left
      # negative value on 0, positive value on 9 make the robot go forward
      # goal = objdetection from PyTorch
      state = (x, y, theta)
      # update_robot_goto(robot, state, goal)
      upper_limit = 2625
      lower_limit = 604
      upper_degree = 30
      lower_degree = -32
      arm_angle = (sensors[0] - lower_limit) / (upper_limit - lower_limit) * (upper_degree - lower_degree) + lower_degree
      update_robot_move_arm(robot, arm_angle, 29)
      update_robot_goto(robot, state, goal_pos)


