import numpy as np
# Map to apriltag poses home setup
# origin is the corner of the balcony, at entrance
import config
config = config.config()
map_apriltag_poses = {}
map_apriltag_poses_home = {}
map_apriltag_poses_bedroom = {}
T = np.identity(4)
T[:3,0] = [1, 0, 0]
T[:3,1] = [0, 0, -1]
T[:3,2] = [0, 1, 0]
T[:3,3] = [50,0, 6.4+config.TAG_SIZE_3IN]
map_apriltag_poses_bedroom[1] = T
T = np.identity(4)
T[:3,0] = [0, 1, 0]
T[:3,1] = [0, 0, -1]
T[:3,2] = [-1, 0, 0]
T[:3,3] = [0, -50, 6.4+config.TAG_SIZE_3IN]
map_apriltag_poses_bedroom[2] = T

# print(T)

for i in range(1,7):
    T = np.identity(4)
    T[:3,1] = np.array([0, 0, -1])
    if i == 1:
        T[:3,0] = [1, 0, 0]
        T[:3,2] = [0, 1, 0]
        T[:3,3] = [41, 197, 26]
    if i == 2:
        T[:3,0] = [1, 0, 0]
        T[:3,2] = [0, 1, 0]
        T[:3,3] = [41+55, 197, 26]
    if i == 3:
        T[:3,0] = [1, 0, 0]
        T[:3,2] = [0, 1, 0]
        T[:3,3] = [41+55+55, 197, 26]
    if i == 4:
        T[:3,0] = [-1, 0, 0]
        T[:3,2] = [0, -1, 0]
        T[:3,3] = [39, 0, 26]
    if i == 5:
        T[:3,0] = [-1, 0, 0]
        T[:3,2] = [0, -1, 0]
        T[:3,3] = [39+55, 0, 26]
    if i == 6:
        T[:3,0] = [-1, 0, 0]
        T[:3,2] = [0, -1, 0]
        T[:3,3] = [39+55+55, 0, 26]
    # print("i = ", i)
    # print(T)
    map_apriltag_poses_home[i] = T




# Map to apriltag poses actual game setup
map_apriltag_poses_game = {}
for i in range(36):
    T = np.identity(4)
    T[:3,1] = np.array([0, 0, -1])
    # apriltag +x is in map +y
    # apriltag +z is in map -x
    if i < 3:
        T[:3,0] = [0, 1, 0]
        T[:3,2] = [-1, 0, 0]
        T[:3,3] = [-72, 6 + 12 * i, 12]
    # apriltag +x is in map +x
    # apriltag +z is in map +y
    elif 3 <= i < 9:
        T[:3,0] = [1, 0, 0]
        T[:3,2] = [0, 1, 0]
        T[:3,3] = [-66 + 12 * (i - 3), 72, 12]
    # apriltag +x is in map -y
    # apriltag +z is in map +x
    elif 9 <= i < 12:
        T[:3,0] = [0, -1, 0]
        T[:3,2] = [1, 0, 0]
        T[:3,3] = [72, 66 - 12 * (i - 9), 12]
    # apriltag +x is in map -x
    # apriltag +z is in map -y
    elif 12 <= i < 18:
        T[:3,0] = [-1, 0, 0]
        T[:3,2] = [0, -1, 0]
        T[:3,3] = [66 - 12 * (i - 12), 0, 12]
    # apriltag +x is in map -y
    # apriltag +z is in map +x
    elif 18 <= i < 21:
        T[:3,0] = [0, -1, 0]
        T[:3,2] = [1, 0, 0]
        T[:3,3] = [72, -6 - 12 * (i - 18), 12]
    # apriltag +x is in map -x
    # apriltag +z is in map -y
    elif 21 <= i < 27:
        T[:3,0] = [-1, 0, 0]
        T[:3,2] = [0, -1, 0]
        T[:3,3] = [66 - 12 * (i - 21), -72, 12]
    # apriltag +x is in map +y
    # apriltag +z is in map -x
    elif 27 <= i < 30:
        T[:3,0] = [0, 1, 0]
        T[:3,2] = [-1, 0, 0]
        T[:3,3] = [-72, -66 + 12 * (i - 27), 12]
    # apriltag +x is in map +x
    # apriltag +z is in map +y
    elif 30 <= i < 36:
        T[:3,0] = [1, 0, 0]
        T[:3,2] = [0, 1, 0]
        T[:3,3] = [-66 + 12 * (i - 30), 0, 12]
    map_apriltag_poses_game[i + 1] = T

if config.FIELD == "HOME":
    map_apriltag_poses = map_apriltag_poses_home
if config.FIELD == "BEDROOM":
    map_apriltag_poses = map_apriltag_poses_bedroom
elif config.FIELD == "GAME":
    map_apriltag_poses = map_apriltag_poses_game
else:
    print("Error: invalid field type")
    exit(1)

if __name__ == "__main__":
    for tagid, pose in map_apriltag_poses.items():
        print("Tag %d:" % tagid)
        print(pose)