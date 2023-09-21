"""
This code is based on Yousef Hamidi's solution to finding position based on apriltags.
We first define the rotation and position of the tags in another file's dictionary,
and then import them, then use a detector to get valid tags and then localize
"""
import numpy as np
import cv2
from pyapriltags import Detector
from scipy.spatial.transform import Rotation

#######################################################################################
# Camera Config
#######################################################################################

T_robot_camera = np.array([
    [0.000, -0.342,  0.940, -14],
    [-1.000,  0.000,  0.000,   0],
    [0.000, -0.940, -0.342,  11],
    [0.000,  0.000,  0.000,   1]
], np.float32)  # camera in reference frame of robot, you will need to calibrate this
T_camera_robot = np.linalg.inv(T_robot_camera)

fx = 460.92495728
fy = 460.85058594
cx = 315.10949707
cy = 176.72598267
camera_params = (fx, fy, cx, cy)

#######################################################################################
# Tag Config
#######################################################################################

tag_size = 3.0  # on a 4in apriltag, only the interior black square is measured

tag_poses = {
    1: np.array([
        [0, 0, -1, -72],
        [1, 0, 0, 24],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], np.float32),
    2: np.array([
        [1, 0, 0, -24],
        [0, 0, 1, 72],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], np.float32),
    3: np.array([
        [1, 0, 0, 24],
        [0, 0, 1, 72],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], np.float32),
    4: np.array([
        [0, 0, 1, 72],
        [-1, 0, 0, 24],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], np.float32),
    5: np.array([
        [0, 0, 1, 72],
        [-1, 0, 0, -24],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], np.float32),
    6: np.array([
        [-1, 0, 0, 24],
        [0, 0, -1, -72],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], np.float32),
    7: np.array([
        [-1, 0, 0, -24],
        [0, 0, -1, -72],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], np.float32),
    8: np.array([
        [0, 0, -1, -72],
        [1, 0, 0, -24],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], np.float32),
}


#######################################################################################
# Apriltag Localization!!!
#######################################################################################

detector = Detector(
    families='tag16h5',
    nthreads=1,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0)


def detect(color_image):
    tags = detector.detect(
        img=cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY),
        estimate_tag_pose=True,
        camera_params=camera_params,
        tag_size=tag_size)

    tags = [tag for tag in tags if tag.decision_margin >
            50 and tag.tag_id in tag_poses.keys()]
    return tags


def Rt2T(R, t):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3:] = t
    return T


def localize(tags):
    # use the formula present in the video, just with the first tag
    if len(tags) == 0:
        return None
    tag = tags[0]
    T_camera_apriltag = Rt2T(tag.pose_R, tag.pose_t)

    T_map_apriltag = tag_poses[tag.tag_id]
    T_apriltag_camera = np.linalg.inv(T_camera_apriltag)
    T_map_robot = T_map_apriltag @ T_apriltag_camera @ T_camera_robot
    return T_map_robot


if __name__ == "__main__":
    # robot = RemoteInterface(...)
    # camera = RealsenseCamera()
    camera = cv2.VideoCapture(0)

    while True:
        # color, depth, sensors = robot.read()
        # color, depth = camera.read()
        _, color = camera.read()
        if color is not None:
            tags = detect(color)
            T = localize(tags)
            if T is not None:  # we found a position!
                x, y = T[0, 3], T[1, 3]
                yaw = Rotation.from_matrix(
                    T[:3, :3]).as_euler("zyx", degrees=True)[0]
                print(x, y, yaw)
            else:
                print("no tags found, so we can't localize")
