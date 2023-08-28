from interface import SimInterface
import numpy as np
import cv2

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

def find_path(objmap, state, goaltag):
  visited = np.zeros(objmap.shape, np.uint8)
  frontier = [(int(state[0]) + 72, int(state[1]) + 72)]
  parent = {}

  while frontier:
    curr = frontier.pop(0)
    visited[curr[0], curr[1]] = True
    if objmap[curr[0], curr[1]] == goaltag:
      path = [curr]
      while curr in parent:
        path.insert(0, parent[curr])
        curr = parent[curr]
      for i in range(len(path)):
        path[i] = (path[i][0] - 72, path[i][1] - 72)
      return path
    neighbors = [
      (curr[0] - 1, curr[1] + 0),
      (curr[0] + 1, curr[1] + 0),
      (curr[0] + 0, curr[1] - 1),
      (curr[0] + 0, curr[1] + 1)
    ]
    for neighbor in neighbors:
      if 0 <= neighbor[0] < 144 and \
         0 <= neighbor[1] < 144 and \
         not visited[neighbor[0], neighbor[1]]:
        frontier.append(neighbor)
        visited[neighbor[0], neighbor[1]] = True
        parent[neighbor] = curr

  return None

if __name__ == "__main__":
  # robot = RemoteInterface("...")
  robot = SimInterface()

  objective_map = np.zeros((144, 144), np.uint8)
  objective_map[:, 66:84] = 1

  while True:
    robot.update()
    # color, depth, sensors = robot.read()
    # x, y, theta = RGBDOdometry()
    sensors = robot.read()
    x, y = robot.pos
    theta = robot.angle # get these from SLAM

    path = find_path(objective_map, (x, y, theta), 1)
    goal = path[min(5, len(path)-1)] # get 5 steps ahead
    update_robot_goto(robot, (x, y, theta), goal)