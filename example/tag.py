# this file can be delteted. It is just for testing the apriltag detection

from datetime import datetime
from io import BytesIO
from multiprocessing import Process, Queue
from pyapriltags import Detector
from scipy.spatial.transform import Rotation as R
import camera
import config 
import cv2
import landmarks
import numpy as np
import socket
import struct
import sys

config = config.Config()
config.TAG_POLICY = "FIRST"
# config.FIELD == "BEDROOM"
config.FIELD == "GAME"


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
                          decode_sharpening=1.2,
                          debug=False)# config.GEN_DEBUG_IMAGES)
    self.camera_params = camera_params
    pass

  def getTagAndPose(self,color, tag_size=config.TAG_SIZE_3IN):
    self.tags = self.at_detector.detect(
      cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), True, self.camera_params[0:4], tag_size)
    if self.tags is not None:
      #get the tag with highest confidence
      max_confidence = config.TAG_DECISION_MARGIN_THRESHOLD
      min_pose_err = config.TAG_POSE_ERROR_THRESHOLD
      max_confidence_tag = None
      for tag in self.tags:
        # print all tag information like pose err and decision margin
        # if tag.tag_id == 6:
          # print(f'tag.decision_margin = {tag.decision_margin}, tag.pose_err = {tag.pose_err}, tag.tag_id = {tag.tag_id}')
          # print(tag.pose_t)
        
        if config.TAG_POLICY == "HIGHEST_CONFIDENCE":
          if tag.decision_margin < config.TAG_DECISION_MARGIN_THRESHOLD and tag.pose_err > config.TAG_POSE_ERROR_THRESHOLD:
            # print(f'tag.decision_margin = {tag.decision_margin} < {config.TAG_DECISION_MARGIN_THRESHOLD}')
            continue
        if tag.decision_margin > max_confidence and tag.pose_err < min_pose_err:
          max_confidence = tag.decision_margin
          min_pose_err = tag.pose_err
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
      # print(f'getRobotPoseFromTagPose: tag_pose = \n{tag_pose}, tag_id = {tag_id}, Lm2Cam = \n{Lm2Cam}')
      return None
    if tag_id in landmarks.map_apriltag_poses:
      # print(f'tag_id = {tag_id} in landmarks.map_apriltag_poses')
      Field2Robot = landmarks.map_apriltag_poses[tag_id] @ np.linalg.inv(Lm2Cam) @ Cam2Robot
      # print(f'landmarks.map_apriltag_poses[{tag_id}] = \n{landmarks.map_apriltag_poses[tag_id]}')
      # print("Lm2Cam = \n", Lm2Cam)
      # print("Cam2Robot = \n", Cam2Robot)
      # print("Field2Robot = \n", Field2Robot)
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
import mask_gen as msk
import matplotlib.pyplot as plt

import socket
import json
def send_command(command, args):
    try:
      print("sending ", command, args)
      client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      client_socket.connect(('localhost', 6000))
      command_data = json.dumps({'command': command, 'args': args})
      client_socket.send(command_data.encode())
      client_socket.close()
    except:
      # print("failed to send", command, args)
      pass
ENABLE_NN = True
if ENABLE_NN:
  import object_detection
ENABLE_OCV = True
import ball_detection_opencv
def main():
  # robot = vex.VexCortex("/dev/ttyUSB0")
  # control = vex.VexControl(robot)
  command_queue = Queue()
  np.set_printoptions(precision=2, suppress=True)
  calib = np.loadtxt("calib.txt", delimiter=",")
  rsCamToRobot = calib[:4,:]
  daiCamToRobot = calib[4:,:]
  camRS = camera.RealSenseCamera(1280,720) 
  camera_paramsRS = camRS.getCameraIntrinsics()
  cam2robotRS = rsCamToRobot
  camDai = camera.DepthAICamera(1920, 1080)
  camera_paramsDai = camDai.getCameraIntrinsics() #1280,720)
  cam2robotDai = daiCamToRobot
  
  atag = ATag(camera_paramsDai)
  # detect the tag and get the pose
  if config.FIELD == "HOME" or config.FIELD == "GAME":
    tag_size = config.TAG_SIZE_3IN # centimeters
  elif config.FIELD == "BEDROOM":
    tag_size = config.TAG_SIZE_6IN
  
  DETECT_ONE_BALL = True
  mask = cv2.bitwise_not(msk.load_mask())
  # convert the mask from color camera reference to depth camera reference
  colorTodepthExtrinsics = camRS.getColorTodepthExtrinsics()
  #invert the extrinsics matrix
  colorTodepthExtrinsics = np.linalg.inv(colorTodepthExtrinsics)
  # warp the mask to the depth camera reference
  mask = cv2.warpPerspective(mask, colorTodepthExtrinsics, (mask.shape[1], mask.shape[0]))
  #apply the mask on the rs depth image and visualize it
  
  objective_map = np.zeros((config.FIELD_HEIGHT, config.FIELD_WIDTH), np.uint8)
  objective_map[:, int(config.FIELD_WIDTH/2 - config.CLAW_LENGTH):int(config.FIELD_WIDTH/2 + config.CLAW_LENGTH) ] = 1
  # Visualizing the objective_map
  plt.figure(figsize=(6, 6))
  plt.imshow(objective_map, cmap='gray')
  plt.title('Objective Map Visualization')
  plt.xlabel('Width')
  plt.ylabel('Height')
  # plt.show()
  ball_map = np.zeros((config.FIELD_HEIGHT, int(config.FIELD_WIDTH/2)), np.uint8)

  while True:
    dt = datetime.now()
    colorDai, depthDai = camDai.read()
    colorRS, depthRS = camRS.read()   
    depthRS = cv2.bitwise_and(depthRS, depthRS, mask=mask)
    #save the images for debugging
    if config.GEN_DEBUG_IMAGES:
      cv2.imwrite("colorDai.jpg", colorDai)
      cv2.imwrite("depthDai.jpg", depthDai)
      cv2.imwrite("colorRS.jpg", colorRS)
      cv2.imwrite("depthRS.jpg", depthRS)
    tagRS, tag_idRS, Lm2CamRS = atag.getTagAndPose(colorRS, tag_size)
    tag, tag_id, Lm2Cam = atag.getTagAndPose(colorDai, tag_size)
    # print(f'Dai tag_id = {tag_idRS}')
    Robot2Field = atag.getRobotPoseFromTagPose(tag, tag_id, Lm2Cam, cam2robotDai)
    # print(f'rs tag_id = {tag_id}')
    Robot2FieldRS = atag.getRobotPoseFromTagPose(tagRS, tag_id, Lm2Cam, cam2robotRS)
    print_robot_pos = False
    if print_robot_pos:
      if Robot2Field is not None:
        print("Robot2Field = ", Robot2Field[0:3,3])
      if Robot2FieldRS is not None:
        print("Robot2FieldRS = ", Robot2FieldRS[0:3,3])

    if colorRS is None or depthRS is None or colorDai is None or depthDai is None:
      continue
    detections = []
    detections_nn = []
    detectionsDai = []
    detections_RS = []
    if ENABLE_NN:
      detections_nn = object_detection.run_object_detection(colorRS)        
    if ENABLE_OCV:
      detections_RS = ball_detection_opencv.ball_detection(colorRS)
      # detectionsDai = ball_detection_opencv.ball_detection(colorDai)
    if len(detections_nn) > 0:
      detections = detections_nn
    elif len(detections_RS) > 0:
      detections = detections_RS

    detectionsDai = [] 
    # print(f'len contours = {len(contours)} , len contoursDai = {len(contoursDai)}')
    SIZE_OF_TENNIS_BALL = 6.54 # centimeters
    ENABLE_RS_BALL_DETECTION = True
    ENABLE_DAI_BALL_DETECTION = False
    if True:
       pass
    if True:
      if len(detections) > 0 and ENABLE_RS_BALL_DETECTION:
          # get the center of the bounding box, and average of the width and height of the bounding box, and calculate the x, y with respect to the field
          # this is the center of the object
          closest_ball_pos_robot = None
          closest_ball_XY = None
          for detection in detections:
            # detection = result[i]
            center_x = detection['center_x']
            center_y = detection['center_y']
            width = detection['width']
            height = detection['height']
            radius = (width + height) / 4
            center = (int(center_x), int(center_y))
            depth_ = depthRS[int(center_y)][int(center_x)]
            # cv2.circle(colorRS, center, int(radius), (0, 255, 0), 2)
            depth_scale = camRS.depth_scale
            depth_ = depth_ * depth_scale

            # get the x, y, z from image frame to camera frame
            x = center_x
            y = center_y
            x = 100 * (x - camRS.cx) * depth_ / camRS.fx  # multiply by 100 to convert to centimeters
            y = 100 * (y - camRS.cy) * depth_ / camRS.fy
            z = 100 * depth_
            
            # print(f'Ball w.r.t to camera x = {x}, y = {y}, z = {z}')
            robot2cam = np.linalg.inv(cam2robotRS)
            ball_pos_robot = robot2cam @ np.array([x, y, z, 1])
            if Robot2Field is not None:
              ball2Field = Robot2Field @ ball_pos_robot
              # update ball_map, making sure that the ball2Field is not out of bounds
              if (ball2Field[0]> 0 and ball2Field[0] < config.FIELD_HEIGHT) and ( ball2Field[1]> 0 and ball2Field[1] < int(config.FIELD_WIDTH/2)):
                ball_map[int(ball2Field[0]), int(ball2Field[1])] = 1
            else:
              continue
            if abs(ball_pos_robot[2]) > 10:
              print("ball position at unusual height ", ball_pos_robot)
              continue
            else:
               cv2.circle(colorRS, center, int(radius), (0, 255, 0), 2)
            if closest_ball_pos_robot is None:
              closest_ball_pos_robot = ball_pos_robot
            else:
              if np.linalg.norm(ball_pos_robot) < np.linalg.norm(closest_ball_pos_robot):
                closest_ball_pos_robot = ball_pos_robot
                closest_ball_XY = ball2Field
            
            theta = np.degrees(np.arctan2(closest_ball_pos_robot[0], closest_ball_pos_robot[1]))
            send_command("rotate", theta)
            if abs(theta) < 30:
               send_command("stop_drive", None) 
      elif len(detectionsDai) and ENABLE_DAI_BALL_DETECTION: 
          #get focal length of dai camera
          fx = camera_paramsDai[0]
          fy = camera_paramsDai[1]
          cx = camera_paramsDai[2]
          cy = camera_paramsDai[3]
          # if the circle's center is not in the center of the image, rotate the robot
          temp_countour = max(detectionsDai, key=lambda x: cv2.contourArea(x))
          (x, y), radius = cv2.minEnclosingCircle(temp_countour)
          center = (int(x), int(y))
          cv2.circle(colorDai, center, int(radius), (0, 255, 0), 2)
          # calculate approx theta
          # print(f'fx = {fx}, fy = {fy}')
          z = 0.5 * SIZE_OF_TENNIS_BALL * fx / radius
          x = (x - cx) * z / fx
          y = (y - cy) * z / fy
          robot2cam = np.linalg.inv(cam2robotDai)
          ball_pos_robot = robot2cam @ np.array([x, y, z, 1])
        # continue
      if tagRS is not None:
        # print the time it took to detect the tag well formatted
        # print("Time to detect tag: ", datetime.now() - dt)
        if tagRS.decision_margin < 50:
            continue
        font_scale = 1 
        font_color = (0, 255, 0)
        font_thickness = 2
        text_offset_y = 30  # Vertical offset between text lines
        # Display tag ID
        #rotate the text by 90 degrees before displaying
        cv2.putText(colorRS, str(tagRS.tag_id), (int(tagRS.center[0]), int(tagRS.center[1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
        # Display only the 1 most significant digit for pose_R and pose_t
        pose_R_single_digit = np.round(tagRS.pose_R[0, 0], 1)  # Round to 1 decimal place
        pose_t_single_digit = np.round(tagRS.pose_t[0], 1)  # Round to 1 decimal place
        cv2.putText(colorRS, str(pose_R_single_digit), (int(tagRS.center[0]), int(tagRS.center[1]) + text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
        cv2.putText(colorRS, str(pose_t_single_digit), (int(tagRS.center[0]), int(tagRS.center[1]) + 2 * text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
        for idx in range(len(tagRS.corners)):
                        cv2.line(colorRS, tuple(tagRS.corners[idx-1, :].astype(int)), tuple(tagRS.corners[idx, :].astype(int)), (0, 255, 0))
        cv2.putText(colorRS, str(tagRS.tag_id),
                    org=(tagRS.center[0].astype(int), tagRS.center[1].astype(int)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255))
      if tag is not None:
        # print the time it took to detect the tag well formatted
        # print("Time to detect tag: ", datetime.now() - dt)
        if tag.decision_margin < 50:
            continue
        font_scale = 0.5  # Adjust this value for a smaller font size
        font_color = (0, 255, 0)
        font_thickness = 2
        text_offset_y = 30  # Vertical offset between text lines
        # Display tag ID
        cv2.putText(colorDai, str(tag.tag_id), (int(tag.center[0]), int(tag.center[1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
        # Display only the 1 most significant digit for pose_R and pose_t
        pose_R_single_digit = np.round(tag.pose_R[0, 0], 1)  # Round to 1 decimal place
        pose_t_single_digit = np.round(tag.pose_t[0], 1)  # Round to 1 decimal place
        cv2.putText(colorDai, str(pose_R_single_digit), (int(tag.center[0]), int(tag.center[1]) + text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
        cv2.putText(colorDai, str(pose_t_single_digit), (int(tag.center[0]), int(tag.center[1]) + 2 * text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
        for idx in range(len(tag.corners)):
                        cv2.line(colorDai, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))
        cv2.putText(colorDai, str(tag.tag_id),
                    org=(tag.center[0].astype(int), tag.center[1].astype(int)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255))
      # stack color images from RS and Dai
      #rotate the  realseanse image
      colorRS = cv2.rotate(colorRS, cv2.ROTATE_90_COUNTERCLOCKWISE)
      resize_scale = colorRS.shape[0] / colorDai.shape[0]
      resized_colorDai = cv2.resize(colorDai, (int(colorDai.shape[1] * resize_scale), colorRS.shape[0]))

      color = np.hstack((colorRS, resized_colorDai))
      # downsize the image for display
      color = cv2.resize(color, (int(color.shape[1] / 2), int(color.shape[0] / 2)))
      cv2.imshow("color", color)
      if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        exit(0)

if __name__ == "__main__":
  main()