import numpy as np
# Map to apriltag poses home setup
# origin is the corner of the balcony, at entrance
import config
config_loc = config.Config()
config_loc.FIELD = "GAME"
map_apriltag_poses = {}
map_apriltag_poses_home = {}
map_apriltag_poses_bedroom = {}
T = np.identity(4)
T[:3,0] = [1, 0, 0]
T[:3,1] = [0, 0, -1]
T[:3,2] = [0, 1, 0]
T[:3,3] = [50,0, 6.4+config_loc.TAG_SIZE_3IN]
map_apriltag_poses_bedroom[1] = T
T = np.identity(4)
T[:3,0] = [0, 1, 0]
T[:3,1] = [0, 0, -1]
T[:3,2] = [-1, 0, 0]
T[:3,3] = [0, -50, 6.4+config_loc.TAG_SIZE_3IN]
map_apriltag_poses_bedroom[2] = T

# print(T)
X_OFFSET = 0 # or 40 cm
for i in range(1,7):
    T = np.identity(4)
    T[:3,1] = np.array([0, 0, -1])
    if i == 1:
        T[:3,0] = [1, 0, 0]
        T[:3,2] = [0, 1, 0]
        T[:3,3] = [X_OFFSET, 197, 26]
    if i == 2:
        T[:3,0] = [1, 0, 0]
        T[:3,2] = [0, 1, 0]
        T[:3,3] = [X_OFFSET+55, 197, 26]
    if i == 3:
        T[:3,0] = [1, 0, 0]
        T[:3,2] = [0, 1, 0]
        T[:3,3] = [X_OFFSET+55+55, 197, 26]
    if i == 4:
        T[:3,0] = [-1, 0, 0]
        T[:3,2] = [0, -1, 0]
        T[:3,3] = [X_OFFSET+55+55, 0, 26]
    if i == 5:
        T[:3,0] = [-1, 0, 0]
        T[:3,2] = [0, -1, 0]
        T[:3,3] = [X_OFFSET+55, 0, 26]
    if i == 6:
        T[:3,0] = [-1, 0, 0]
        T[:3,2] = [0, -1, 0]
        T[:3,3] = [X_OFFSET, 0, 26]
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

import numpy as np

northmap_apriltag_poses = {}
southmap_apriltag_poses = {}
for i in range(18):
    T = np.identity(4)
    T[:3,1] = np.array([0, 0, -1])
    # apriltag +x is in map +y
    # apriltag +z is in map -x
    if i < 3:
        T[:3,0] = [0, 1, 0]
        T[:3,2] = [-1, 0, 0]
        T[:3,3] = [-72, 12 + 24 * i, 12]
    # apriltag +x is in map +x
    # apriltag +z is in map +y
    elif 3 <= i < 9:
        T[:3,0] = [1, 0, 0]
        T[:3,2] = [0, 1, 0]
        T[:3,3] = [-60 + 24 * (i - 3), 72, 12]
    # apriltag +x is in map -y
    # apriltag +z is in map +x
    elif 9 <= i < 12:
        T[:3,0] = [0, -1, 0]
        T[:3,2] = [1, 0, 0]
        T[:3,3] = [72, 60 - 24 * (i - 9), 12]
    # apriltag +x is in map -x
    # apriltag +z is in map -y
    elif 12 <= i < 18:
        T[:3,0] = [-1, 0, 0]
        T[:3,2] = [0, -1, 0]
        T[:3,3] = [60 - 24 * (i - 12), 0, 12]
    northmap_apriltag_poses[i] = T
for i in range(12, 30):
    T = np.identity(4)
    T[:3,1] = np.array([0, 0, -1])
    # apriltag +x is in map +x
    # apriltag +z is in map +y
    if 12 <= i < 18:
        T[:3,0] = [1, 0, 0]
        T[:3,2] = [0, 1, 0]
        T[:3,3] = [60 - 24 * (i - 12), 0, 12]
    # apriltag +x is in map +y
    # apriltag +z is in map -x
    if 18 <= i < 21:
        T[:3,0] = [0, 1, 0]
        T[:3,2] = [-1, 0, 0]
        T[:3,3] = [-72, -12 - 24 * (i - 18), 12]
    # apriltag +x is in map -x
    # apriltag +z is in map -y
    elif 21 <= i < 27:
        T[:3,0] = [-1, 0, 0]
        T[:3,2] = [0, -1, 0]
        T[:3,3] = [-60 + 24 * (i - 21), -72, 12]
    # apriltag +x is in map -y
    # apriltag +z is in map +x
    elif 27 <= i < 30:
        T[:3,0] = [0, -1, 0]
        T[:3,2] = [1, 0, 0]
        T[:3,3] = [72, -60 + 24 * (i - 27), 12]
    southmap_apriltag_poses[i] = T

# if __name__ == "__main__":
#     for tagid, pose in northmap_apriltag_poses.items():
#         print("North Tag %d:" % tagid)
#         print(pose)
#     for tagid, pose in southmap_apriltag_poses.items():
#         print("South Tag %d:" % tagid)
#         print(pose)
map_apriltag_poses = northmap_apriltag_poses
# if config_loc.FIELD == "HOME":
#     map_apriltag_poses = map_apriltag_poses_home
# if config_loc.FIELD == "BEDROOM":
#     map_apriltag_poses = map_apriltag_poses_bedroom
# elif config_loc.FIELD == "GAME":
#     map_apriltag_poses = map_apriltag_poses_game
# else:
#     print("Error: invalid field type")
#     exit(1)
def convert_inch_to_cm(T):
    """
    Convert the x, y, z values in the transformation matrix from inches to centimeters.

    Parameters:
    T (numpy.ndarray): A 4x4 transformation matrix where the last column of the first
                       three rows represent the x, y, z coordinates in inches.

    Returns:
    numpy.ndarray: The transformed matrix with x, y, z values in centimeters.
    """
    # Create a copy of the matrix to avoid modifying the original matrix
    T_converted = T.copy()
    
    # Conversion factor from inches to centimeters
    inches_to_cm = 2.54
    
    # Converting x, y, z values from inches to centimeters
    T_converted[:3, 3] *= inches_to_cm
    
    return T_converted

for tagid, pose in map_apriltag_poses.items():
    map_apriltag_poses[tagid] = convert_inch_to_cm(pose)

if __name__ == "__main__":
    for tagid, pose in map_apriltag_poses.items():
        map_apriltag_poses[tagid] = convert_inch_to_cm(pose)
    # for tagid, pose in map_apriltag_poses.items():
        print("Tag %d:" % tagid)
        print(pose)