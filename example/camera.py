import camera
import time

####################################################################
import numpy as np
import pyrealsense2 as rs
from typing import Tuple
import cv2
from datetime import datetime
from pyapriltags import Detector
from scipy.spatial.transform import Rotation as R

#!/usr/bin/env python3

#For tiny yolo
from pathlib import Path
import sys
import depthai as dai

'''
#    Translation Vector: 0.00552000012248755  -0.00510000018402934  -0.011739999987185  

# Extrinsic from "Accel"	  To	  "Color" :
#  Rotation Matrix:
#    0.99982          0.0182547        0.00521857    
#   -0.0182514        0.999833        -0.000682113   
#   -0.00523015       0.000586744      0.999986   
'''
class RealSenseCamera:
  def __init__(self, width=640, height=360, debug_mode=True):
    self.debug_mode = debug_mode
    self.initCamera(width, height)

  def initCamera(self, width, height):
    self.width = width
    self.height = height
    self.shape = (height, width)
    self.pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
    profile = self.pipeline.start(config)
    # intel realsense on jetson nano sometimes get misdetected as 2.1 even though it has 3.2 USB
    # profile = self.pipeline.start()
    depth_sensor = profile.get_device().first_depth_sensor()
    self.depth_scale = depth_sensor.get_depth_scale()
    self.align = rs.align(rs.stream.color)
    # Get the intrinsics of the color stream
    self.color_sensor = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile()
    self.color_intrinsics = self.color_sensor.get_intrinsics()
    self.fx = self.color_intrinsics.fx
    self.fy = self.color_intrinsics.fy
    self.cx = self.color_intrinsics.ppx
    self.cy = self.color_intrinsics.ppy
    self.hfov = np.degrees(np.arctan2(self.width  / 2, self.fx)) * 2
    self.vfov = np.degrees(np.arctan2(self.height / 2, self.fy)) * 2
    self.hfov_rad = np.radians(self.hfov)
    self.vfov_rad = np.radians(self.vfov)
    self.hfov_rad_half = self.hfov_rad / 2
    self.vfov_rad_half = self.vfov_rad / 2
    #TODO Get camera to IMU extrinsics
    #depth filtering
    self.min_depth = 0.1
    self.max_depth = 3.0 
    
  def resetCamera(self, width, height):
        # Don't do hw reset as it slows down the camera initialization
    try:
      # Create a context object
      ctx = rs.context()
      # Get a device
      dev = ctx.devices[0]
      # Reset the camera
      dev.hardware_reset()
      self.initCamera(width, height)
    except:
      print("No realsense camera found, or failed to reset.")

  def getCameraIntrinsics(self):
    return (self.fx, self.fy, self.cx, self.cy, self.width, self.height, self.depth_scale)
  
  def capture(self) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    Grab frames from the realsense camera
    Args:
        unit (str, optional): inches|meters|raw. Defaults to 'inches'.
    Returns:
        Tuple[bool, np.ndarray, np.ndarray]: status, color image, depth image
    """
    if self.pipeline:
      try:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return True, color_image, depth_image
      except:
        if self.pipeline:
          self.pipeline.stop()
          self.pipeline = None

    return False, None, None

  def __del__(self):
    if self.pipeline:
      self.pipeline.stop()
      self.pipeline = None
      # delete cv2 window if it exists
      try:
        cv2.destroyAllWindows()
      except:
        pass



  def read(self, scale=False): # will also store in buffer to be read
    ret, color, depth = self.capture()
    if not ret:
      return None, None #np.zeros((self.height, self.width, 3), dtype=np.uint8), \
            #  np.zeros((self.height, self.width), dtype=np.float32)
    if scale:
      depth = depth.astype(np.float32) * self.depth_scale
    # color = np.ascontiguousarray(np.flip(color, axis=-1))

    return color, depth

  def depth2rgb(self, depth):
    if depth.dtype == np.uint16:
      return cv2.applyColorMap(np.sqrt(depth).astype(np.uint8), cv2.COLORMAP_JET)
    else:
      return cv2.applyColorMap(np.floor(np.sqrt(depth / self.depth_scale)).astype(np.uint8), cv2.COLORMAP_JET)

  def view(self):
    color, depth = self.read()
    combined = np.hstack((color, self.depth2rgb(depth)))
    cv2.imshow("combined", combined)
    if cv2.waitKey(1) == 27:
      exit(0)
  
  def viewFilteredDepth(self):
    color, depth = self.read()
    depth = depth.astype(np.float32) * self.depth_scale
    mask = np.bitwise_and(depth > self.min_depth, depth < self.max_depth)
    filtered_depth = np.where(mask, depth, 0)
    filtered_color = np.where(np.tile(mask.reshape(self.height, self.width, 1), (1, 1, 3)), color, 0)
    combined = np.hstack((filtered_color, self.depth2rgb(depth), self.depth2rgb(filtered_depth)))
    cv2.imshow("filtered_color, depth, and filtered depth", combined)
    if cv2.waitKey(1) == 27:
      exit(0)

  def viewDepth(self):
    color, depth = self.read()
    cv2.imshow("depth", self.depth2rgb(depth))
    if cv2.waitKey(1) == 27:
      exit(0)

  def ViewColor(self):
    color, depth = self.read()
    cv2.imshow("color", color)
    if cv2.waitKey(1) == 27:
      exit(0)

  def getFilteredColorBasedOnDepth(self):
    color, depth = self.read()
    depth = depth.astype(np.float32) * self.depth_scale
    mask = np.bitwise_and(depth > self.min_depth, depth < self.max_depth)
    self.filtered_color = np.where(np.tile(mask.reshape(self.height, self.width, 1), (1, 1, 3)), color, 0)
    # cv2.imshow("filtered color", filtered_color)
    # if cv2.waitKey(1) == 27:
    #   exit(0)

  def getTagPose(self,  tag_size, specified_tag = 255, input_type="color"):
    at_detector = Detector(families='tag16h5',
                           nthreads=4,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=1.2,
                           debug=0)
    camera_params = self.getCameraIntrinsics()
    # detect the tag and get the pose
    while True:
      if input_type == "color":
        color, depth = self.read()
        self.filtered_color = color    # TODO vairable cleanup 
      elif input_type == "filtered_color":
        self.getFilteredColorBasedOnDepth()
      # check if the coloe image is valid
      if self.filtered_color is None:
        continue
      tags = at_detector.detect(
      cv2.cvtColor(self.filtered_color, cv2.COLOR_BGR2GRAY), True, camera_params[0:4], tag_size)
      if specified_tag != 255:
        for tag in tags:
          if tag.tag_id == specified_tag:
            R = tag.pose_R
            t = tag.pose_t
            # make 4 * 4 transformation matrix
            T = np.eye(4)
            T[0:3, 0:3] = R
            T[0:3, 3] = t.ravel()
            return T
      else:
        for tag in tags:
          if tag.decision_margin > 150: 
            return tags
          break
        return None

  def hackYdist(self, tag_size, input_type="color"): # other input type is filtered_color
      tags = self.getTagPose(tag_size=tag_size)
      if tags is not None:
        # for tag in tags:
        #   cv2.imshow("color", self.filtered_color)
        #   if cv2.waitKey(1) == 27:
        #     cv2.destroyAllWindows()
        #     exit(0)
        #   continue
        for tag in tags:
            if tag.decision_margin < 150:
                continue
            # print the tag pose, tag id, etc well formatted
            # print("Tag Family: ", tag.tag_family)
            print("Tag ID: ", tag.tag_id)
            # print("Tag Hamming Distance: ", tag.hamming)
            # print("Tag Decision Margin: ", tag.decision_margin)
            # print("Tag Homography: ", tag.homography)
            # print("Tag Center: ", tag.center)
            # print("Tag Corners: ", tag.corners)
            # print("Tag Pose: ", tag.pose_R, tag.pose_t)
            # print("Tag Pose Error: ", tag.pose_err)
            # print("Tag Size: ", tag.tag_size)
            print("Tag y distance: ", tag.pose_t[2])
            return tag.pose_t[2] # return y distance Hack
        
  def visTagPose(self, tag_size, input_type="color"): # other input type is filtered_color
      tags = self.getTagPose(tag_size=tag_size)
      if tags is not None:
        # for tag in tags:
        #   cv2.imshow("color", self.filtered_color)
        #   if cv2.waitKey(1) == 27:
        #     cv2.destroyAllWindows()
        #     exit(0)
        #   continue
        for tag in tags:
            if tag.decision_margin < 150:
                continue
            # print the tag pose, tag id, etc well formatted
            # print("Tag Family: ", tag.tag_family)
            print("Tag ID: ", tag.tag_id)
            # print("Tag Hamming Distance: ", tag.hamming)
            # print("Tag Decision Margin: ", tag.decision_margin)
            # print("Tag Homography: ", tag.homography)
            # print("Tag Center: ", tag.center)
            # print("Tag Corners: ", tag.corners)
            # print("Tag Pose: ", tag.pose_R, tag.pose_t)
            # print("Tag Pose Error: ", tag.pose_err)
            # print("Tag Size: ", tag.tag_size)
            print("Tag y distance: ", tag.pose_t[2])
            font_scale = 0.5  # Adjust this value for a smaller font size
            font_color = (0, 255, 0)
            font_thickness = 2
            text_offset_y = 30  # Vertical offset between text lines
            # Display tag ID
            cv2.putText(self.filtered_color, str(tag.tag_id), (int(tag.center[0]), int(tag.center[1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
            # Display only the 1 most significant digit for pose_R and pose_t
            pose_R_single_digit = np.round(tag.pose_R[0, 0], 1)  # Round to 1 decimal place
            pose_t_single_digit = np.round(tag.pose_t[0], 1)  # Round to 1 decimal place
            cv2.putText(self.filtered_color, str(pose_R_single_digit), (int(tag.center[0]), int(tag.center[1]) + text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(self.filtered_color, str(pose_t_single_digit), (int(tag.center[0]), int(tag.center[1]) + 2 * text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
        cv2.imshow("color", self.filtered_color)
        if cv2.waitKey(1) == 27:
          cv2.destroyAllWindows()
          exit(0)

  def getColor(self):
    color, depth = self.read()
    return color

  def calibrateCameraWrtLandMark(self,tag_size=5, viz=False):
    # make 4 * 4 transformation matrix
    T = np.eye(4)
    camera_params = self.getCameraIntrinsics()
    at_detector = Detector(families='tag16h5',
                          nthreads=1,
                          quad_decimate=1.0,
                          quad_sigma=0.0,
                          refine_edges=1,
                          decode_sharpening=1,
                          debug=0)
    # detect the tag and get the pose
    count = 0
    while True:
      count += 1
      color, depth = self.read()
      tags = at_detector.detect(
        cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), True, camera_params[0:4], tag_size=tag_size)
      found_tag = False
      for tag in tags:
          if tag.decision_margin < 50: 
            continue
          found_tag = True
          if tag.tag_id != 0:
            continue
          # print("Tag ID: ", tag.tag_id)
          # print("Tag Pose: ", tag.pose_R, tag.pose_t)
          R = tag.pose_R
          t = tag.pose_t
          T[0:3, 0:3] = R
          T[0:3, 3] = t.ravel()
          # print("Tag Pose Error: ", tag.pose_err)
          # print("Tag Size: ", tag.tag_size)
      if not viz:
        if count > 10:
          return T
        else:
          continue
      else:
        # np.set_printoptions(precision=2, suppress=True)
        print(T)
        if not found_tag:
          cv2.imshow("color", color)
          if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            exit(0)
          continue
        for tag in tags:
            if tag.tag_id != 0:
                continue
            if tag.decision_margin < 50:
                continue
            font_scale = 0.5  # Adjust this value for a smaller font size
            font_color = (0, 255, 0)
            font_colorR = (0, 0, 255)
            font_thickness = 2
            text_offset_y = 30  # Vertical offset between text lines
            # Display tag ID
            cv2.putText(color, str(tag.tag_id), (int(tag.center[0]), int(tag.center[1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
            # Display only the 1 most significant digit for pose_R and pose_t
            # pose_R_single_digit = np.round(tag.pose_R[0, 0], 1)  # Round to 1 decimal place
            pose_tx_single_digit = np.round(tag.pose_t[0], 1)  # Round to 1 decimal place
            pose_ty_single_digit = np.round(tag.pose_t[1], 1)  # Round to 1 decimal place
            pose_tz_single_digit = np.round(tag.pose_t[2], 1)  # Round to 1 decimal place
            # create a string for the pose T  and R
            pose_t_single_digit = np.array2string(tag.pose_t, formatter={'float_kind':lambda x: "%.1f" % x})
            pose_R_single_digit = np.array2string(tag.pose_R, formatter={'float_kind':lambda x: "%.1f" % x})
            cv2.putText(color, str(pose_R_single_digit), (int(tag.center[0]), int(tag.center[1]) + text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(color, str(pose_t_single_digit), (int(tag.center[0]), int(tag.center[1]) + 2 * text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_colorR, font_thickness, cv2.LINE_AA)
        cv2.imshow("color", color)
        if cv2.waitKey(1) == 27:
          cv2.destroyAllWindows()
          exit(0)

  def calibrateCameraWrtRobot(self,tag_size=5):
      #TransformationLandMarkToRobot 
      # For L2R I hand measured the april tag distance from the robot base. Robot base is the center of the robot's claw
      self.L2R = np.array([[1, 0, 0, -5], [0, -1, 0, 47.7], [0, 0, -1, 0], [0, 0, 0, 1]])
      #TransformationLmToCamera 
      # self.L2C= np.array([           [ 0.98008357,  0.19856912, 0.00255036, -2.67309145 ],           [-0.04210883,  0.19525296, 0.97984852, 8.52999655  ],           [ 0.19406969, -0.96044083, 0.19972573, 64.82263177 ],          [0,            0,          0,          1           ] ])
      self.L2C= self.calibrateCameraWrtLandMark(tag_size=tag_size, viz=False)
      return  np.linalg.inv(self.L2C) @ self.L2R

  def getDistance(self) :#, C2R, L2C):
    return self.getTagPose(7.62, 0) @ self.calibrateCameraWrtRobot(7.62) @ np.array([0,0,0,1])

#####################################################################################

from vex_serial import VexCortex
from enum import Enum
class clawAction(Enum):
  Open = 1
  Close = -1


def drive_forward(robot, value, drive_time=1, left_motor=0, right_motor=9):
  motor_values = robot.motor
  left_drive = 1
  right_drive = -1
  motor_values[left_motor] = left_drive * value
  motor_values[right_motor] = right_drive * value
  robot.motors(motor_values)
  time.sleep(drive_time)
  stop_drive(robot)

def drive_backward(robot, value, drive_time=1, left_motor=0, right_motor=9):
  motor_values = robot.motor
  left_drive = -1
  right_drive = 1
  motor_values[left_motor] = left_drive * value
  motor_values[right_motor] = right_drive * value
  time.sleep(drive_time)
  stop_drive(robot)

def stop_drive(robot):
  motor_values = 10*[0]
  robot.motors(motor_values)



def claw(robot, value, action = clawAction.Close, claw_motor=1):
  motor_values = robot.motor
  if action == clawAction.close:
    motor_values[claw_motor] = -1 * value
  else:
    motor_values[claw_motor] = -1 * value
  robot.motors(motor_values)
  stop_drive()


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
  
def rotateRobot(robot, seconds, dir, speed):
  robot.motor[0] = speed * dir#if theta > 0 else -127
  robot.motor[9] = speed * dir #if theta > 0 else -127
  time.sleep(seconds)
  robot.motor[0] = 0 #if theta > 0 else -127
  robot.motor[9] = 0 #if theta > 0 else -127


def testRotate(robot, rot_speed=40, rot_time=5):
  rot_dir  = 1
  rotateRobot(robot, rot_time, rot_dir, rot_speed)
  time.sleep(0.5)
  rotateRobot(robot, rot_time, -rot_dir, rot_speed)
  time.sleep(0.5)



def testAngle(robot):
    # goal_pos = np.random.rand(2) * 50
    # goal angle between -180 and 180
    goal_angle = np.random.rand() * 360 - 180
    update_robot_goto(x, y, goal_angle)
    update_robot_goto(x, y, -goal_angle)

def testTransulateAlongY(robot, cam):
  set_Y = 200
  numOfIterations = 50
  # while True:
  for i in range(numOfIterations):
    # cam.view()
    # cam.getFilteredColorBasedOnDepth()
    # cam.visTagPose(tag_size=7.62)
    Y = cam.hackYdist(tag_size=7.62)
    if Y is None:
      continue
    # print(cam.calibrateCameraWrtRobot(tag_size=5))
    # print(cam.getDistance())
    if robot.running():
      if Y > set_Y:
        drive_forward(robot,30)
        stop_drive(robot)
      else:
        drive_backward(robot,30)
        stop_drive(robot)
#####################################################################################

    
class CalibrateCamera:
 def __init__(self, cam):
    self.camera_params = [cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height]
    self.cam = cam
    return
 def calibrateCameraWrtRobot(self,tag_size=5):
      #TransformationLandMarkToRobot 
      # For L2R I hand measured the april tag distance from the robot base. Robot base is the center of the robot's claw
      # self.L2R = np.array([[1, 0, 0, -5], [0, -1, 0, 47.7], [0, 0, -1, 0], [0, 0, 0, 1]])
      self.L2R = np.array([[1, 0, 0, -5], [0, -1, 0, 47.7], [0, 0, -1, 0], [0, 0, 0, 1]])
      #TransformationLmToCamera 
      # self.L2C= np.array([           [ 0.98008357,  0.19856912, 0.00255036, -2.67309145 ],           [-0.04210883,  0.19525296, 0.97984852, 8.52999655  ],           [ 0.19406969, -0.96044083, 0.19972573, 64.82263177 ],          [0,            0,          0,          1           ] ])
      self.L2C= self.calibrateCameraWrtLandMark(tag_size=tag_size, viz=False)
      return  np.linalg.inv(self.L2C) @ self.L2R
 
 def calibrateCameraWrtLandMark(self,tag_size=5, viz=False):
    # make 4 * 4 transformation matrix
    T = np.eye(4)
    at_detector = Detector(families='tag16h5',
                          nthreads=1,
                          quad_decimate=1.0,
                          quad_sigma=0.0,
                          refine_edges=1,
                          decode_sharpening=1,
                          debug=0)
    # detect the tag and get the pose
    count = 0
    while True:
      count += 1
      self.color, depth = self.cam.read()
      tags = at_detector.detect(
        cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY), True, self.camera_params[0:4], tag_size=tag_size)
      found_tag = False
      for tag in tags:
          if tag.decision_margin < 50: 
            continue
          found_tag = True
          if tag.tag_id != 0:
            continue
          # print("Tag ID: ", tag.tag_id)
          # print("Tag Pose: ", tag.pose_R, tag.pose_t)
          R = tag.pose_R
          t = tag.pose_t
          T[0:3, 0:3] = R
          T[0:3, 3] = t.ravel()
          # print("Tag Pose Error: ", tag.pose_err)
          # print("Tag Size: ", tag.tag_size)
      if not viz:
        if count > 10:
          return T
        else:
          continue
      else:
        # np.set_printoptions(precision=2, suppress=True)
        # print(T)
        if not found_tag:
          cv2.imshow("color", self.color)
          if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            return T
            # exit(0)
          continue
        for tag in tags:
            if tag.tag_id != 0:
                continue
            if tag.decision_margin < 50:
                continue
            font_scale = 0.5  # Adjust this value for a smaller font size
            font_color = (0, 255, 0)
            font_thickness = 2
            text_offset_y = 30  # Vertical offset between text lines
            # Display tag ID
            cv2.putText(self.color, str(tag.tag_id), (int(tag.center[0]), int(tag.center[1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
            # Display only the 1 most significant digit for pose_R and pose_t
            pose_R_single_digit = np.round(tag.pose_R[0, 0], 1)  # Round to 1 decimal place
            pose_t_single_digit = np.round(tag.pose_t[0], 1)  # Round to 1 decimal place
            cv2.putText(self.color, str(pose_R_single_digit), (int(tag.center[0]), int(tag.center[1]) + text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
            cv2.putText(self.color, str(pose_t_single_digit), (int(tag.center[0]), int(tag.center[1]) + 2 * text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
        cv2.imshow("color", self.color)
        if cv2.waitKey(1) == 27:
          cv2.destroyAllWindows()
          return T
          # exit(0)





class DepthAICamera:
  def __init__(self, width=1920, height=1080, object_detection=False, debug_mode=True):
    self.debug_mode = debug_mode
    self.initCamera(width, height, object_detection)

   

  def initCamera(self, width, height, object_detection):
    self.camera_params = self.getCameraIntrinsics(width, height)
    # Create pipeline
    self.pipeline = dai.Pipeline()
    # Define source and output
    if not object_detection:
      self.camRgb = self.pipeline.create(dai.node.ColorCamera)
      self.camRgb.setPreviewSize(width, height)
      # self.camRgb.initialControl.setManualFocus(130)

      self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)
      self.xoutRgb.setStreamName("rgb_default")
      self.camRgb.preview.link(self.xoutRgb.input)

      # self.camLeft = self.pipeline.create(dai.node.MonoCamera)
      # self.camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
      # self.camLeft.setCamera("left")
      # self.xoutLeft = self.pipeline.create(dai.node.XLinkOut)
      # self.xoutLeft.setStreamName("left")
      # self.camLeft.out.link(self.xoutLeft.input)

      # Connect to device and start pipeline
      self.device = dai.Device(self.pipeline)
      self.qRgb = self.device.getOutputQueue(name="rgb_default")
      # self.qLeft = self.device.getOutputQueue(name="left")

      #warm up the camera to get rid of the first few frames that have low exposure
      for i in range(10):
        self.qRgb.get()
    else:
      self.labelMap = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
    ]
      self.syncNN = True
      # Get argument first
      self.nnBlobPath = str((Path(__file__).parent / Path('/home/nvidia/wsp/clawbot/depthai-python/examples/models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
      if not Path(self.nnBlobPath).exists():
          raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')
      # Create pipeline
      # Define sources and outputs
      self.camRgb = self.pipeline.create(dai.node.ColorCamera)
      spatialDetectionNetwork = self.pipeline.create(dai.node.YoloSpatialDetectionNetwork)
      monoLeft = self.pipeline.create(dai.node.MonoCamera)
      monoRight = self.pipeline.create(dai.node.MonoCamera)
      stereo = self.pipeline.create(dai.node.StereoDepth)
      nnNetworkOut = self.pipeline.create(dai.node.XLinkOut)

      xoutRgb = self.pipeline.create(dai.node.XLinkOut)
      xoutNN = self.pipeline.create(dai.node.XLinkOut)
      xoutDepth = self.pipeline.create(dai.node.XLinkOut)

      xoutRgb.setStreamName("rgb")
      xoutNN.setStreamName("detections")
      xoutDepth.setStreamName("depth")
      nnNetworkOut.setStreamName("nnNetwork")

      # Properties
      self.camRgb.setPreviewSize(416, 416)
      self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
      self.camRgb.setInterleaved(False)
      self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

      monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
      monoLeft.setCamera("left")
      monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
      monoRight.setCamera("right")

      # setting node configs
      stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
      # Align depth map to the perspective of RGB camera, on which inference is done
      stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
      stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
      stereo.setSubpixel(True)

      spatialDetectionNetwork.setBlobPath(self.nnBlobPath)
      spatialDetectionNetwork.setConfidenceThreshold(0.5)
      spatialDetectionNetwork.input.setBlocking(False)
      spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
      spatialDetectionNetwork.setDepthLowerThreshold(100)
      spatialDetectionNetwork.setDepthUpperThreshold(5000)

      # Yolo specific parameters
      spatialDetectionNetwork.setNumClasses(80)
      spatialDetectionNetwork.setCoordinateSize(4)
      spatialDetectionNetwork.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
      spatialDetectionNetwork.setAnchorMasks({ "side26": [1,2,3], "side13": [3,4,5] })
      spatialDetectionNetwork.setIouThreshold(0.5)

      # Linking
      monoLeft.out.link(stereo.left)
      monoRight.out.link(stereo.right)

      self.camRgb.preview.link(spatialDetectionNetwork.input)
      spatialDetectionNetwork.passthrough.link(xoutRgb.input)

      spatialDetectionNetwork.out.link(xoutNN.input)

      stereo.depth.link(spatialDetectionNetwork.inputDepth)
      spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
      spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)



  def getCameraIntrinsics(self, width, height):
    self.width = width
    self.height = height
    with dai.Device() as device:
      calibFile = str((Path(__file__).parent / Path(f"calib_{device.getMxId()}.json")).resolve().absolute())
      if len(sys.argv) > 1:
          calibFile = sys.argv[1]

      calibData = device.readCalibration()
      calibData.eepromToJsonFile(calibFile)

      M_rgb, wid, hgt = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.CAM_A)
      self.cx = M_rgb[0][2] * (self.width / wid)
      self.cy = M_rgb[1][2] * (self.height / hgt)
      self.fx = M_rgb[0][0] * (self.width / wid)
      self.fy = M_rgb[1][1] * (self.height / hgt)
      if self.debug_mode:
        print("OAKD-1 RGB Camera Default intrinsics...")
        print(M_rgb)
        print(wid)
        print(hgt)
      # assert wid == self.width and hgt == self.height

      return [self.fx, self.fy, self.cx, self.cy]

  def read(self, scale=False): # will also store in buffer to be read
    imgFrame = self.qRgb.get()
    color_image = imgFrame.getCvFrame()
    # imgFrame = self.qLeft.get()
    # left_image = imgFrame.getCvFrame()
    # if scale:
    #   depth = depth.astype(np.float32) * self.depth_scale
    # color = np.ascontiguousarray(np.flip(color, axis=-1))
    return color_image, None
  
  def runObjectDetection(self):
    # Connect to device and start pipeline
    with dai.Device(self.pipeline) as device:

        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        networkQueue = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False);

        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)
        printOutputLayersOnce = True

        while True:
            inPreview = previewQueue.get()
            inDet = detectionNNQueue.get()
            depth = depthQueue.get()
            inNN = networkQueue.get()

            if printOutputLayersOnce:
                toPrint = 'Output layer names:'
                for ten in inNN.getAllLayerNames():
                    toPrint = f'{toPrint} {ten},'
                print(toPrint)
                printOutputLayersOnce = False;

            frame = inPreview.getCvFrame()
            depthFrame = depth.getFrame() # depthFrame values are in millimeters

            depth_downscaled = depthFrame[::4]
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
            max_depth = np.percentile(depth_downscaled, 99)
            depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

            counter+=1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time

            detections = inDet.detections

            # If the frame is available, draw bounding boxes on it and show the frame
            height = frame.shape[0]
            width  = frame.shape[1]
            for detection in detections:
                # if label is not a sports ball or orange, continue
                # get sports ball and orange label from labelMap
                orage_label = self.labelMap.index("orange")
                sports_ball_label = self.labelMap.index("sports ball")
                if detection.label != sports_ball_label and detection.label != orage_label:
                    continue
                roiData = detection.boundingBoxMapping
                roi = roiData.roi
                roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                topLeft = roi.topLeft()
                bottomRight = roi.bottomRight()
                xmin = int(topLeft.x)
                ymin = int(topLeft.y)
                xmax = int(bottomRight.x)
                ymax = int(bottomRight.y)
                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 1)

                # Denormalize bounding box
                x1 = int(detection.xmin * width)
                x2 = int(detection.xmax * width)
                y1 = int(detection.ymin * height)
                y2 = int(detection.ymax * height)
                try:
                    label = self.labelMap[detection.label]
                except:
                    label = detection.label
                cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
            cv2.imshow("depth", depthFrameColor)
            cv2.imshow("rgb", frame)

            if cv2.waitKey(1) == ord('q'):
                break

  
def vizDaicam():
  # Create pipeline
  pipeline = dai.Pipeline()
  # Define source and output
  camRgb = pipeline.create(dai.node.ColorCamera)
  #set preview size to max possible resolution for this camera
  camRgb.setPreviewSize(1280, 720)
  # fix the focus so that we can get a clear image
  camRgb.initialControl.setManualFocus(130)

  xoutRgb = pipeline.create(dai.node.XLinkOut)
  xoutRgb.setStreamName("rgb")
  camRgb.preview.link(xoutRgb.input)

  camLeft = pipeline.create(dai.node.MonoCamera)
  camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
  camLeft.setCamera("left")

  xoutLeft = pipeline.create(dai.node.XLinkOut)
  xoutLeft.setStreamName("left")
  camLeft.out.link(xoutLeft.input)

  # Connect to device and start pipeline
  with dai.Device(pipeline) as device:
      qRgb = device.getOutputQueue(name="rgb")
      qLeft = device.getOutputQueue(name="left")
    
      while True:
          txt = ""
          for q in [qRgb, qLeft]:
              imgFrame = q.get()
              name = q.getName()
              txt += f"[{name}] Exposure: {imgFrame.getExposureTime().total_seconds()*1000:.3f} ms, "
              txt += f"ISO: {imgFrame.getSensitivity()},"
              txt += f" Lens position: {imgFrame.getLensPosition()},"
              txt += f" Color temp: {imgFrame.getColorTemperature()} K   "
              cv2.imshow(name, imgFrame.getCvFrame())
          print(txt)
          if cv2.waitKey(1) == ord('q'):
              break

  # def calibrateCameraWrtRobot(self,tag_size=5):
  #     #TransformationLandMarkToRobot 
  #     # For L2R I hand measured the april tag distance from the robot base. Robot base is the center of the robot's claw
  #     self.L2R = np.array([[1, 0, 0, -5], [0, -1, 0, 47.7], [0, 0, -1, 0], [0, 0, 0, 1]])
  #     #TransformationLmToCamera 
  #     # self.L2C= np.array([           [ 0.98008357,  0.19856912, 0.00255036, -2.67309145 ],           [-0.04210883,  0.19525296, 0.97984852, 8.52999655  ],           [ 0.19406969, -0.96044083, 0.19972573, 64.82263177 ],          [0,            0,          0,          1           ] ])
  #     self.L2C= self.calibrateCameraWrtLandMark(tag_size=tag_size, viz=False)
  #     return  np.linalg.inv(self.L2C) @ self.L2R


rsCam2Robot = np.array([
 [ 9.94469395e-01,  8.00882118e-02,  6.79448348e-02, -1.87907316e+01],
 [-5.13571977e-02, -1.93490083e-01,  9.79757126e-01,  6.56196051e+01],
 [ 9.16136479e-02, -9.77827933e-01, -1.88306860e-01,  2.59744550e+01],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
 ])

# rsCam2LandMark =
# [[ 9.92912028e-01  1.09941141e-01  4.51514199e-02  1.47855808e+01]
#  [-1.06699686e-01  9.91905905e-01 -6.88320647e-02  2.12833899e+00]
#  [-5.23534358e-02  6.35265426e-02  9.96605988e-01  6.34547258e+01]
#  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
# if the file is run directly
if __name__ == "__main__":
  np.set_printoptions(precision=2, suppress=True)
  # rsCam = RealSenseCamera(1280, 720)
  # rsCamCalib = CalibrateCamera(rsCam)
  # rsCam2LM  = rsCamCalib.calibrateCameraWrtLandMark(tag_size=7.62, viz=True)
  # print("rsCam2LM = \n", rsCam2LM)
  # rsCamToRobot =   rsCamCalib.calibrateCameraWrtRobot(tag_size=5)
  # print("rsCamToRobot = \n", rsCamToRobot)

  object_detection = True
  if not object_detection:
    daiCam = DepthAICamera(1280,720, object_detection=False)
    daiCamCalib = CalibrateCamera(daiCam)
    daiCam2LM = daiCamCalib.calibrateCameraWrtLandMark(tag_size=7.62, viz=True)
    print("daiCam2LM = \n", daiCam2LM)
    daiCamToRobot =   daiCamCalib.calibrateCameraWrtRobot(tag_size=5)
    print("daiCamToRobot = \n", daiCamToRobot)
  else:
    daiCam = DepthAICamera(1280,720, object_detection=True)
    daiCam.runObjectDetection()

  # daiCam2Robot = daiCam2LM @ np.linalg.inv(rsCam2LM) @ rsCam2Robot

  # print("daiCam2Robot = \n", daiCam2Robot)

  # cam.resetCamera(1280, 720)
  # robot = VexCortex("/dev/ttyUSB0")
  # testTransulateAlongY(robot, cam)

# testRotate(robot)
# robot.motor[0] = 0 #if theta > 0 else -127
# robot.motor[9] = 0 #if theta > 0 else -127
