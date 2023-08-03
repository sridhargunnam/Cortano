from cortano import RemoteInterface

if __name__ == "__main__":
  robot = RemoteInterface("1.2.3.4")
  while True:
    robot.update() # must never be forgotten
    color, depth, sensors = robot.read()

    forward = robot.keys["w"] - robot.keys["s"]
    robot.motor[0] =  forward * 127
    robot.motor[9] = -forward * 127
