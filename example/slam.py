import open3d
import camera
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyapriltags import Detector
from datetime import datetime
import config

def get_pose(R, t):
  T = np.identity(4)
  T[:3,:3] = R
  T[:3,3] = t.reshape(3)
  return T


"""
function to get the robot pose from the tag pose, 
the tags are located. 
Robots will be autonomously competing 1v1 on a 12 feet x 12 feet field. 
The field will be separated in half with a wall, with each adversary on opposite sides of the field. 
Robots are allowed to start in any position in their starting pose, but cannot start moving until the remote network sends a Ready signal (this will be verified before each round starts). 
At the start of the round, 30 tennis balls are placed on each side of the field in an undisclosed symmetric formation, to prevent pre-programming.
On the field, 4 inch x 4 inch AprilTags will be placed every 4 feet from each other (with exception of corners) on each side. These tags can be used to help with localization. 

Here are the AprilTag locations (family tag16h5) in inches, where (0, 0) is the center of the field
tag_id: 1, location: (-72, 24), orientation: (1, 0)
tag_id: 2, location: (-24, 72), orientation: (0, -1)
tag_id: 3, location: (24, 72), orientation: (0, -1)
tag_id: 4, location: (72, 24), orientation: (-1, 0)
tag_id: 5, location: (72, -24), orientation: (-1, 0)
tag_id: 6, location: (24, -72), orientation: (0, 1)
tag_id: 7, location: (-24, -72), orientation: (0, 1)
tag_id: 8, location: (-72, 24), orientation: (1, 0)
"""
def get_robot_pose(tag_pose, tag_id):
  if tag_id == 1:
    robot_x = tag_pose[0] - 72
    robot_y = tag_pose[1] + 24
    



















# def get_robot_pose(tag_pose, tag_id, tag_orientation):
#   if tag_id == 1:
#     robot_pose = tag_pose + np.array([72, -24, 0])
#     robot_orientation = tag_orientation
#   elif tag_id == 2:
#     robot_pose = tag_pose + np.array([24, -72, 0])
#     robot_orientation = tag_orientation
#   elif tag_id == 3:
#     robot_pose = tag_pose + np.array([-24, -72, 0])
#     robot_orientation = tag_orientation
#   elif tag_id == 4:
#     robot_pose = tag_pose + np.array([-72, -24, 0])
#     robot_orientation = tag_orientation
#   elif tag_id == 5:
#     robot_pose = tag_pose + np.array([-72, 24, 0])
#     robot_orientation = tag_orientation
#   elif tag_id == 6:
#     robot_pose = tag_pose + np.array([-24, 72, 0])
#     robot_orientation = tag_orientation
#   elif tag_id == 7:
#     robot_pose = tag_pose + np.array([24, 72, 0])
#     robot_orientation = tag_orientation
#   elif tag_id == 8:
#     robot_pose = tag_pose + np.array([72, 24, 0])
#     robot_orientation = tag_orientation
#   else:
#     print("Invalid tag id")
#     return None
#   return robot_pose, robot_orientation

class ATag:
  def __init__(self, camera_params) -> None:
    self.at_detector = Detector(families='tag16h5',
                          nthreads=1,
                          quad_decimate=1.0,
                          quad_sigma=0.0,
                          refine_edges=1,
                          decode_sharpening=1,
                          debug=0)
    self.camera_params = camera_params
    pass

  def getTagAndPose(self,color):
    self.tags = self.at_detector.detect(
      cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), True, self.camera_params[0:4], config.TAG_SIZE)
    if tags is not None:
      #get the tag with highest confidence
      max_confidence = 0
      max_confidence_tag = None
      for tag in tags:
        if tag.decision_margin < config.TAG_DECISION_MARGIN_THRESHOLD: 
          continue
        if tag.decision_margin > max_confidence:
          max_confidence = tag.decision_margin
          max_confidence_tag = tag
      if max_confidence_tag is not None:
        tag_pose = max_confidence_tag.pose_t
        tag_id = max_confidence_tag.tag_id
        return tag_pose, tag_id
    return None, None
  
def getRobotPoseFromTagPose(tag_pose, tag_id):
    robot_pos, robot_orientation = get_robot_pose(tag_pose, tag_id)
    return robot_pos, robot_orientation

class Open3DSlam:
  def __init__(self, cam) -> None:
    self.option = open3d.pipelines.odometry.OdometryOption()
    self.camera_params =  cam.getCameraParams()
    self.fx = self.camera_params[0]
    self.fy = self.camera_params[1]
    self.cx = self.camera_params[2]
    self.cy = self.camera_params[3]
    self.width  = self.camera_params[4]
    self.height = self.camera_params[5]
    self.cam_intrinsic_params = open3d.camera.PinholeCameraIntrinsic(self.width, self.height, self.fx, self.fy, self.cx, self.cy)
    self.prev_rgbd_image = None

  def get_o3d_pose(self, cam):
    color, depth = cam.read()
    # filter the depth image so that noise is removed
    depth = depth.astype(np.float32) / 1000.
    mask = np.bitwise_and(depth > 0.1, depth < 3.0) # -> valid depth points
    filtered_depth = np.where(mask, depth, 0)
    # converting to Open3D's format so we can do odometry
    o3d_color = open3d.geometry.Image(color)
    o3d_depth = open3d.geometry.Image(filtered_depth)
    o3d_rgbd  = open3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth)
    if self.prev_rgbd_image is not None:
      self.prev_rgbd_image = o3d_rgbd
      return np.identity(4)
    is_success, T, _ = open3d.pipelines.odometry.compute_rgbd_odometry(
      o3d_rgbd, self.prev_rgbd_image, self.cam_intrinsic_params, T,
      open3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), self.option)
    if is_success:
      return T
    return np.identity(4)


if __name__ == "__main__":
  # robot = RemoteInterface("...")
  config = config()
  cam = camera.RealSenseCamera()
  o3dSlam = Open3DSlam(cam)
  atag = ATag(cam.getCameraParams())
  global_T = np.identity(4)
  while True:   
    found_tag = False
    # tags = at_detector.detect(
    #   cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), True, camera_params[0:4], tag_size=7.62)
    tags = cam.getTagPose(tag_size=config.TAG_SIZE)
    robot_pos , robot_orientation = getRobotPoseFromTagPose(tags)

    if tags is not None:
      for tag in tags:
          if tag.decision_margin < 150: 
            continue
          found_tag = True
          global_T = get_pose(tag.pose_R, tag.pose_t) # use your get_pose algorithm here!
          totalTagPoses += 1
          break
    
    if not found_tag and prev_rgbd_image is not None: # use RGBD odometry relative transform to estimate pose
      T = np.identity(4)
      ret, T, _ = open3d.pipelines.odometry.compute_rgbd_odometry(
        o3d_rgbd, prev_rgbd_image, cam_intrinsic_params, T,
        open3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
      global_T = global_T.dot(T)
      rotation = global_T[:3,:3]
      # print("Rotation: ", R.from_matrix(rotation).as_rotvec(degrees=True))
      totalSlamPoses += 1
    prev_rgbd_image = o3d_rgbd # we forgot this last time!

    # dont need this, but helpful to visualize
    filtered_color = np.where(np.tile(mask.reshape(height, width, 1), (1, 1, 3)), color, 0)
    cv2.imshow("color", filtered_color)
    if cv2.waitKey(1) == 27:
      et = datetime.now() 
      tt = (et-st).total_seconds()
      print("total time = ", tt)
      print("totalTagPoses", totalTagPoses)
      print("totalSlamPoses", totalSlamPoses)
      print("fps = ", ( (totalTagPoses + totalSlamPoses) / tt ) )
      exit(0)
    # process_time = datetime.now() - dt
    # print("FPS: " + str(1 / process_time.total_seconds()))

    # Use custom_draw_geometry_with_camera_trajectory to visualize the camera trajectory
    # and geometry. Press 'Q' to exit.
    

