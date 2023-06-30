from interface import RemoteInterface

if __name__ == "__main__":
  robot = RemoteInterface("<ip.address.here>")
  while True:
    forward = robot.keys["w"] - robot.keys["s"]
    robot.motor[0] = forward * 64
    robot.motor[9] = forward * 64
    robot.update()