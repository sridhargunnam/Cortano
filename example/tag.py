# this file can be delteted. It is just for testing the apriltag detection
import camera
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyapriltags import Detector
from datetime import datetime
import sys
import config 
import landmarks
config = config.config()
config.TAG_POLICY = "FIRST"
config.FIELD == "BEDROOM"

def readCalibrationFile(path=config.CALIB_PATH):
  calib = np.loadtxt(path, delimiter=",")
  rsCamToRobot = calib[:4,:]
  daiCamToRobot = calib[4:,:]
  return rsCamToRobot, daiCamToRobot

class ATag:
  def __init__(self, camera_params) -> None:
    self.at_detector = Detector(families='tag16h5',
                          nthreads=1,
                          quad_decimate=1.0,
                          quad_sigma=0.0,
                          refine_edges=1,
                          decode_sharpening=1,
                          debug=config.GEN_DEBUG_IMAGES)
    self.camera_params = camera_params
    pass

  def getTagAndPose(self,color, tag_size=config.TAG_SIZE_3IN):
    self.tags = self.at_detector.detect(
      cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), True, self.camera_params[0:4], tag_size)
    if self.tags is not None:
      #get the tag with highest confidence
      max_confidence = 0
      max_confidence_tag = None
      for tag in self.tags:
        if config.TAG_POLICY == "HIGHEST_CONFIDENCE":
          if tag.decision_margin < config.TAG_DECISION_MARGIN_THRESHOLD: 
            print(f'tag.decision_margin = {tag.decision_margin} < {config.TAG_DECISION_MARGIN_THRESHOLD}')
            continue
        if tag.decision_margin > max_confidence:
          max_confidence = tag.decision_margin
          max_confidence_tag = tag
      if max_confidence_tag is not None:
        # tag_pose = max_confidence_tag.pose_t
        tag_id = max_confidence_tag.tag_id
        R = max_confidence_tag.pose_R
        t = max_confidence_tag.pose_t
        # make 4 * 4 transformation matrix
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t.ravel()
        return max_confidence_tag, tag_id, T
    return None, None, None
  
  
  def getRobotPoseFromTagPose(self, tag_pose, tag_id, Lm2Cam, Cam2Robot):
    if tag_pose is None or tag_id is None or Lm2Cam is None:
      print(f'getRobotPoseFromTagPose: tag_pose = \n{tag_pose}, tag_id = {tag_id}, Lm2Cam = \n{Lm2Cam}')
      return None
    if tag_id in landmarks.map_apriltag_poses:
      Field2Robot = landmarks.map_apriltag_poses[tag_id] @ np.linalg.inv(Lm2Cam) @ Cam2Robot
      print("Field2Robot = \n", Field2Robot)
    # np.savetxt("debug_transformation.txt", np.vstack((Lm2Cam, Cam2Robot)), delimiter=",")
      return Field2Robot
    else:
      return None


"""
function to get the robot pose from the tag pose, 
the tags are located. 
Robots will be autonomously competing 1v1 on a 12 feet x 12 feet field. 
The field will be separated in half with a wall, with each adversary on opposite sides of the field. 
Robots are allowed to start in any position in their starting pose, but cannot start moving until the remote network sends a Ready signal (this will be verified before each round starts). 
At the start of the round, 30 tennis balls are placed on each side of the field in an undisclosed symmetric formation, to prevent pre-programming.
On the field, 4 inch x 4 inch AprilTags will be placed every 4 feet from each other (with exception of corners) on each side. These tags can be used to help with localization. 
"""

def main():
  np.set_printoptions(precision=2, suppress=True)
  #check the input arguments and set the camera
  #rscam for realsense camera, and daicam for dai camera
  calib = np.loadtxt("calib.txt", delimiter=",")
  rsCamToRobot = calib[:4,:]
  daiCamToRobot = calib[4:,:]
  if len(sys.argv) != 2:
    print("Usage: python3 tag.py [rscam/daicam]")
    # exit(0)
  if sys.argv[1] == "rscam":
    cam = camera.RealSenseCamera(1280,720) 
    camera_params = cam.getCameraIntrinsics()
    cam2robot = rsCamToRobot
  elif sys.argv[1] == "daicam":
    cam = camera.DepthAICamera(1280,720)
    camera_params = cam.getCameraIntrinsics(1280,720)
    cam2robot = daiCamToRobot
  atag = ATag(camera_params)
  # detect the tag and get the pose
  if config.FIELD == "HOME" or config.FIELD == "GAME":
    tag_size = config.TAG_SIZE_3IN # centimeters
  elif config.FIELD == "BEDROOM":
    tag_size = config.TAG_SIZE_6IN

  while True:
    dt = datetime.now()
    color, depth = cam.read()    
    tag, tag_id, Lm2Cam = atag.getTagAndPose(color, tag_size)
    
    Robot2Field = atag.getRobotPoseFromTagPose(tag, tag_id, Lm2Cam, cam2robot)

    if tag is not None:
      robot2Lm = np.linalg.inv(Lm2Cam) @ rsCamToRobot
      Lm2Robot = np.linalg.inv(robot2Lm)
      # print("Lm2Robot = \n", Lm2Robot)
      # print the time it took to detect the tag well formatted
      print("Time to detect tag: ", datetime.now() - dt)
      if tag.decision_margin < 50:
          continue
      font_scale = 0.5  # Adjust this value for a smaller font size
      font_color = (0, 255, 0)
      font_thickness = 2
      text_offset_y = 30  # Vertical offset between text lines
      # Display tag ID
      cv2.putText(color, str(tag.tag_id), (int(tag.center[0]), int(tag.center[1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
      # Display only the 1 most significant digit for pose_R and pose_t
      pose_R_single_digit = np.round(tag.pose_R[0, 0], 1)  # Round to 1 decimal place
      pose_t_single_digit = np.round(tag.pose_t[0], 1)  # Round to 1 decimal place
      cv2.putText(color, str(pose_R_single_digit), (int(tag.center[0]), int(tag.center[1]) + text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
      cv2.putText(color, str(pose_t_single_digit), (int(tag.center[0]), int(tag.center[1]) + 2 * text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
    cv2.imshow("color", color)
    if cv2.waitKey(1) == 27:
      cv2.destroyAllWindows()
      exit(0)

if __name__ == "__main__":
  main()