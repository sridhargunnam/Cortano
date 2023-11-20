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
import logging
#!/usr/bin/env python3
import config

#For tiny yolo
from pathlib import Path
import sys
import depthai as dai

import queue
# create queue of size 10 
daiQueue = queue.Queue(100)

from vex_serial import VexCortex 

class RealSenseCamera:
  def __init__(self, width=640, height=360, fps=30, debug_mode=True):
    self.debug_mode = debug_mode
    self.initCamera(width, height, fps)

  def initCamera(self, width, height, fps):
    self.width = width
    self.height = height
    self.shape = (height, width)
    self.pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
    profile = self.pipeline.start(config)
    # intel realsense on jetson nano sometimes get misdetected as 2.1 even though it has 3.2 USB
    # profile = self.pipeline.start()
    depth_sensor = profile.get_device().first_depth_sensor()
    self.depth_scale = depth_sensor.get_depth_scale()
    self.align = rs.align(rs.stream.color)
    # Get the intrinsics of the color stream
    self.color_sensor = self.pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile()
    self.color_intrinsics = self.color_sensor.get_intrinsics()
    self.colorTodepthExtrinsics = self.color_sensor.get_extrinsics_to(profile.get_stream(rs.stream.depth))  
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

  def getColorTodepthExtrinsics(self):
    # Get the extrinsics from color to depth
    extrinsics = self.color_sensor.get_extrinsics_to(self.pipeline.get_active_profile().get_stream(rs.stream.depth))
    # Construct the transformation matrix (3x4)
    R = np.array(extrinsics.rotation).reshape(3, 3)
    T = np.array(extrinsics.translation).reshape(3, 1)

    # Use only the rotation part for the homography matrix
    # and assume the translation in z-axis does not significantly affect the transformation
    homography_matrix = np.eye(3, dtype=np.float64)
    homography_matrix[:2, :2] = R[:2, :2]  # Using only the top-left 2x2 part of the rotation matrix
    homography_matrix[:2, 2] = T[:2, 0]    # Using the x and y components of the translation vector

    return homography_matrix

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
    try:
      if self.pipeline:
        self.pipeline.stop()
        self.pipeline = None
        # delete cv2 window if it exists
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
 
def TransformationInverse(T):
   R = T[:3,:3]
   Ri = R.T
   t = T[:3,3:]
   ti = Ri @ -t
   Ti = np.eye(4)
   Ti[:3,:3] = Ri
   Ti[:3,3:] = ti
   return Ti
    
class CalibrateCamera:
 def __init__(self, cam):
    self.camera_params = [cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height]
    self.cam = cam
    return

 def getCamera2Robot(self,tag_size=5, tag_id=0, viz = False):
      #TransformationLandMarkToRobot 
      # For L2R I hand measured the april tag distance from the robot base. Robot base is the center of the robot's claw
      if tag_size == 5:
        self.L2R = np.array([[1, 0, 0, 5], [0, -1, 0, -47.7], [0, 0, -1, 0], [0, 0, 0, 1]])
      elif tag_size == 12.7:
        tag_y = (22.94 + 16.3) # 11 inch to 22.94 cm 
        self.L2R = np.array([[1, 0, 0, 0], [0, -1, 0, tag_y], [0, 0, -1, 0], [0, 0, 0, 1]])
      print(" L2R = \n", self.L2R)
      self.C2L= self.calibrateCameraWrtLandMark(tag_size=tag_size, tag_id=tag_id, viz=viz)
      print("self.C2L = \n", self.C2L)
      # return  TransformationInverse(self.C2L @ self.L2R) 
      #Note L2R = R2L upon writing it down manually 
      return  (self.C2L @ self.L2R) 
      # return  TransformationInverse(self.L2R @ self.C2L) 
 
 def calibrateCameraWrtLandMark(self,tag_size=5, tag_id=0, viz=False):
    # make 4 * 4 transformation matrix
    cfg = config.Config()
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
    print("self.camera_params[0:4] = ", self.camera_params[0:4])
    while True:
      count += 1
      self.color, depth = self.cam.read()
      tags = at_detector.detect(
        cv2.cvtColor(self.color, cv2.COLOR_BGR2GRAY), True, self.camera_params[0:4], tag_size=tag_size)
      found_tag = False
      for tag in tags:
          if tag.decision_margin < cfg.TAG_DECISION_MARGIN_THRESHOLD and tag.pose_err > cfg.TAG_POSE_ERROR_THRESHOLD: 
            continue
          found_tag = True
          # if tag.tag_id != 0:
          #   continue
          if tag.tag_id != tag_id:
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
            if tag.tag_id != tag_id:
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


# from vex_serial import VexControl
class DepthAICamera:
  def __init__(self, width=1920, height=1080, object_detection=False, viz= False,  debug_mode=False):
    self.debug_mode = debug_mode
    self.initCamera(width, height, object_detection)
    self.viz = viz
    # self.robot = VexCortex("/dev/ttyUSB0")
    # self.control = VexControl(self.robot)
    #Create a queue to store the timestamp of the frame, x,y,z position of the detected object, and the confidence score
    # self.q = daiQueue
  # descructor to stop the pipeline
  def __del__(self):
    try:
      cv2.destroyAllWindows()
    except:
      pass
    # if self.robot:
    #   self.robot.stop()

  def initCamera(self, width, height, object_detection):
    self.width = width
    self.height = height
    # Create pipeline
    self.pipeline = dai.Pipeline()
    # Define source and output
    if not object_detection: # rgb only
      print("initilying depthai color and depth")
      # Closer-in minimum depth, disparity range is doubled (from 95 to 190):
      extended_disparity = False
      # Better accuracy for longer distance, fractional disparity 32-levels:
      subpixel = True
      # Better handling for occlusions:
      lr_check = True
      self.camRgb = self.pipeline.create(dai.node.ColorCamera)
      self.camRgb.setPreviewSize(width, height)
      self.camRgb.setInterleaved(True)
      self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

      self.StereoDepth = self.pipeline.create(dai.node.StereoDepth)
      self.spatialLocationCalculator = self.pipeline.create(dai.node.SpatialLocationCalculator)


      # self.camRgb.initialControl.setManualFocus(130)

      self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)
      self.xoutRgb.setStreamName("rgb_default")
      self.camRgb.preview.link(self.xoutRgb.input)
      
      self.camLeft = self.pipeline.create(dai.node.MonoCamera)
      self.camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
      self.camLeft.setCamera("left")
      self.xoutLeft = self.pipeline.create(dai.node.XLinkOut)
      self.xoutLeft.setStreamName("left")
      self.camLeft.out.link(self.xoutLeft.input)

      self.camRight = self.pipeline.create(dai.node.MonoCamera)
      self.camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
      self.camRight.setCamera("right")
      self.xoutRight = self.pipeline.create(dai.node.XLinkOut)
      self.xoutRight.setStreamName("right")
      self.camRight.out.link(self.xoutRight.input)

      # Link left and right cameras to StereoDepth node
      self.camLeft.out.link(self.StereoDepth.left)
      self.camRight.out.link(self.StereoDepth.right)

      self.xoutDisparity = self.pipeline.create(dai.node.XLinkOut)
      self.xoutDisparity.setStreamName("disparity")
      self.xoutSpatialData = self.pipeline.create(dai.node.XLinkOut)
      self.xinSpatialCalcConfig = self.pipeline.create(dai.node.XLinkIn)

      self.xoutSpatialData.setStreamName("spatialData")
      self.xinSpatialCalcConfig.setStreamName("spatialCalcConfig")


      # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
      self.StereoDepth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
      # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
      self.StereoDepth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
      self.StereoDepth.setLeftRightCheck(lr_check)
      self.StereoDepth.setExtendedDisparity(extended_disparity)
      self.StereoDepth.setSubpixel(subpixel)
      # self.spatialLocationCalculator.inputConfig.setWaitForMessage(False)
      self.StereoDepth.disparity.link(self.xoutDisparity.input)
      # self.StereoDepth.depth.link(self.spatialLocationCalculator.inputDepth)


      # Connect to device and start pipeline
      self.device = dai.Device(self.pipeline)
      self.qRgb = self.device.getOutputQueue(name="rgb_default", maxSize=2, blocking=False)
      self.qLeft = self.device.getOutputQueue(name="left")
      self.qRight = self.device.getOutputQueue(name="right")
      self.qDepth = self.device.getOutputQueue(name="disparity", maxSize=2, blocking=False)

      self.camera_params = self.getCameraIntrinsics() #width, height)

      self.depth_scale = self.fx


      '''      # Config
      topLeft = dai.Point2f(0.4, 0.4)
      bottomRight = dai.Point2f(0.6, 0.6)

      config = dai.SpatialLocationCalculatorConfigData()
      config.depthThresholds.lowerThreshold = 100
      config.depthThresholds.upperThreshold = 10000
      calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MEDIAN
      config.roi = dai.Rect(topLeft, bottomRight)

# 
      self.spatialLocationCalculator.inputConfig.setWaitForMessage(False)
      self.spatialLocationCalculator.initialConfig.addROI(config)'''
      # self.spatialLocationCalculator.passthroughDepth.link(self.xoutDisparity.input)
      # stereo.depth.link(self.spatialLocationCalculator.inputDepth)

      # self.spatialLocationCalculator.out.link(self.xoutSpatialData.input)
      # self.xinSpatialCalcConfig.out.link(self.spatialLocationCalculator.inputConfig)
      #warm up the camera to get rid of the first few frames that have low exposure
      for i in range(10):
        if self.qRgb.has():
          self.qRgb.get()
        # self.qLeft.get()
        # self.qRight.get()
        if self.qDepth.has():
          self.qDepth.get()
    # else: #objection detection part starts from here
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
      self.spatialDetectionNetwork = self.pipeline.create(dai.node.YoloSpatialDetectionNetwork)
      self.monoLeft = self.pipeline.create(dai.node.MonoCamera)
      self.monoRight = self.pipeline.create(dai.node.MonoCamera)
      self.stereo = self.pipeline.create(dai.node.StereoDepth)
      self.nnNetworkOut = self.pipeline.create(dai.node.XLinkOut)

      self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)
      self.xoutNN = self.pipeline.create(dai.node.XLinkOut)
      self.xoutDepth = self.pipeline.create(dai.node.XLinkOut)

      self.xoutRgb.setStreamName("rgb")
      self.xoutNN.setStreamName("detections")
      self.xoutDepth.setStreamName("depth")
      self.nnNetworkOut.setStreamName("nnNetwork")
      # Properties
      self.camRgb.setPreviewSize(416, 416)
      self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
      self.camRgb.setInterleaved(False)
      self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

      self.monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
      self.monoLeft.setCamera("left")
      self.monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
      self.monoRight.setCamera("right")

      # setting node configs
      self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
      # Align depth map to the perspective of RGB camera, on which inference is done
      self.stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
      self.stereo.setOutputSize(self.monoLeft.getResolutionWidth(), self.monoLeft.getResolutionHeight())
      self.stereo.setSubpixel(True)

      self.spatialDetectionNetwork.setBlobPath(self.nnBlobPath)
      self.spatialDetectionNetwork.setConfidenceThreshold(0.5)
      self.spatialDetectionNetwork.input.setBlocking(False)
      self.spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
      self.spatialDetectionNetwork.setDepthLowerThreshold(100)
      self.spatialDetectionNetwork.setDepthUpperThreshold(5000)

      # Yolo specific parameters
      self.spatialDetectionNetwork.setNumClasses(80)
      self.spatialDetectionNetwork.setCoordinateSize(4)
      self.spatialDetectionNetwork.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
      self.spatialDetectionNetwork.setAnchorMasks({ "side26": [1,2,3], "side13": [3,4,5] })
      self.spatialDetectionNetwork.setIouThreshold(0.5)

      # Linking
      self.monoLeft.out.link(self.stereo.left)
      self.monoRight.out.link(self.stereo.right)

      self.camRgb.preview.link(self.spatialDetectionNetwork.input)
      self.spatialDetectionNetwork.passthrough.link(self.xoutRgb.input)

      self.spatialDetectionNetwork.out.link(self.xoutNN.input)

      self.stereo.depth.link(self.spatialDetectionNetwork.inputDepth)
      self.spatialDetectionNetwork.passthroughDepth.link(self.xoutDepth.input)
      self.spatialDetectionNetwork.outNetwork.link(self.nnNetworkOut.input)
    
      # Output queues will be used to get the rgb frames and nn data from the outputs defined above
      # self.previewQueue      = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
      # self.detectionNNQueue  = self.device.getOutputQueue(name="detections", maxSize=4, blocking=False)
      # self.depthQueue        = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
      # self.networkQueue      = self.device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False);
      # self.runObjectDetection()

  def getCameraIntrinsics(self):#, width=1920, height=1080):
    calibData = self.device.readCalibration()
    import math 
    M_rgb, wid, hgt = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.CAM_A)
    self.cx = M_rgb[0][2] * (self.width / wid)
    self.cy = M_rgb[1][2] * (self.height / hgt)
    self.fx = M_rgb[0][0] * (self.width / wid)
    self.fy = M_rgb[1][1] * (self.height / hgt)
    self.focal_length_in_pixels = self.fx
    self.hfov =  2 * 180 / (math.pi) * math.atan(self.width * 0.5 / self.fx)
    self.vfov =  2 * 180 / (math.pi) * math.atan(self.height * 0.5 / self.fy)
    self.baseline = 7.5 #cm
    if True: #self.debug_mode:
      print("OAKD-1 RGB Camera Default intrinsics...")
      print(M_rgb)
      print(wid)
      print(hgt)
      print("cx = ", self.cx)
      print("cy = ", self.cy)
      print("fx = ", self.fx)
      print("fy = ", self.fy)
      # exit()
    # assert wid == self.width and hgt == self.height

    return [self.fx, self.fy, self.cx, self.cy]

  def read(self, scale=False): # will also store in buffer to be read
    # print("read from camera")
    color = self.qRgb.get().getCvFrame()
    if self.qDepth.has():
      depth = self.qDepth.get().getFrame()
    else:
      depth = np.zeros((self.height, self.width), dtype=np.float32)
    # imgFrame = self.qLeft.get()
    # left_image = imgFrame.getCvFrame()
    # if scale:
    #   depth = depth.astype(np.float32) * self.depth_scale
    # color = np.ascontiguousarray(np.flip(color, axis=-1))
    return color, depth 
  
  def view(self):
    while True:
      color, _ = self.read()
      # view color and depth image using opencv
      cv2.imshow("color", self.color)
      # cv2.imshow("depth", depth)
      if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        exit(0)
      continue
  def getTagPoses(self,tag_size=5, viz=True):
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
    print("self.camera_params[0:4] = ", self.camera_params[0:4])
    while True:
      count += 1
      color = self.read()
      tags = at_detector.detect(
        cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), True, self.camera_params[0:4], tag_size=tag_size)
      found_tag = False
      for tag in tags:
          if tag.decision_margin < 50: 
            continue
          found_tag = True
          # if tag.tag_id != 0:
          #   continue
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
          cv2.imshow("color", color)
          if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            return T
            # exit(0)
          continue
        for tag in tags:
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
          return T
          # exit(0)
  def runObjectDetection(self):
    # Connect to device and start pipeline
    # with dai.Device(self.pipeline) as device:


        globalTime = time.monotonic()
        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)
        printOutputLayersOnce = True

        while True:
        # if True:
            inPreview = self.previewQueue.get()
            inDet = self.detectionNNQueue.get()
            depth = self.depthQueue.get()
            inNN = self.networkQueue.get()

            if printOutputLayersOnce:
                toPrint = 'Output layer names:'
                for ten in inNN.getAllLayerNames():
                    toPrint = f'{toPrint} {ten},'
                print(toPrint)
                printOutputLayersOnce = False;

            frame = inPreview.getCvFrame()
            # print(inPreview.getTimestamp())
            # print(inPreview.getSequenceNum())
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
            count = 0
            for detection in detections:
                if detection.confidence < 1:
                    #print detection confidence
                    continue
                else:
                   print("Confidence: " + str(detection.confidence))
                # if label is not a sports ball or orange, continue
                # get sports ball and orange label from labelMap
                orage_label = self.labelMap.index("orange")
                sports_ball_label = self.labelMap.index("sports ball")
                if detection.label != sports_ball_label and detection.label != orage_label:
                    continue
                #Add the timestamp and the x,y,z position of the detected object to the queue
                # if size of queue is greater than 10, pop the oldest element
                # if daiQueue.qsize() > 100:
                #   print("Queue is full, exiting")
                #   exit(0)
                # else:
                #   daiQueue.put([inPreview.getTimestamp(), int(detection.spatialCoordinates.x), int(detection.spatialCoordinates.y), int(detection.spatialCoordinates.z), detection.confidence])
                #   time.sleep(1)
                # print ([inPreview.getTimestamp(), int(detection.spatialCoordinates.x), int(detection.spatialCoordinates.y), int(detection.spatialCoordinates.z), detection.confidence])
                # print(f"X: {int(detection.spatialCoordinates.x)} mm")
                # print(f"Y: {int(detection.spatialCoordinates.y)} mm")
                # print(f"Z: {int(detection.spatialCoordinates.z)} mm")
                # print(f"Confidence: {detection.confidence}")
                print(f"time: {inPreview.getTimestamp()}")
                if self.debug_mode:
                  print(f"X: {int(detection.spatialCoordinates.x/10)} cm")
                  print(f"Y: {int(detection.spatialCoordinates.y/10)} cm")
                  print(f"Z: {int(detection.spatialCoordinates.z/10)} cm")
                self.ballX = int(detection.spatialCoordinates.x) / 10
                self.ballY = int(detection.spatialCoordinates.z) / 10
                self.ballTheta = abs(np.degrees(np.arctan2(self.ballY,  self.ballX)))
                # robot_state = [0,0, 0]
                # goal_state = [self.ballX, self.ballY, 0]
                # self.control.update_robot_goto([0,0], [self.ballX, self.ballY])  
                # self.goToGoalPosition()
                count += 1
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
                cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x/10)} cm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y/10)} cm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z/10)} cm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
            if self.viz:
              cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
              cv2.imshow("depth", depthFrameColor)
              cv2.imshow("rgb", frame)
            print(f"Number of sports ball and orange detected: {count}")

            if cv2.waitKey(1) == ord('q'):
              # self.control.drive_backward(40, 4)
              self.robot.stop()
              exit(0)

  # TODO hack untill multiprocess is working
  def goToGoalPosition(self):
       print("going to self.ballY = ", self.ballY)
  #  robot = VexCortex("/dev/ttyUSB0")
  #  while True:
       if abs(self.ballY) < 5:
        self.control.stop_drive()
        return
       if self.robot.running():
        # get the timestamp, x,y,z position, and confidence score from the queue
        # if the queue is empty, it will throw an exception
        if self.ballY > 5:
          self.control.drive_forward(30)
          self.control.stop_drive()
        else:
          self.control.drive_backward(30)
          self.control.stop_drive()
       else:
        print("robot is not running")
        # time.sleep(1)
  
import config
cfg = config.Config()
# SIZE_OF_CALIBRATION_TAG = 5 #cm

def runCameraCalib(input="Load"):
  np.set_printoptions(precision=2, suppress=True)
  if input == "calib_setup1": # dai and rs camera have common field of vie wand are looking at the same tag
      print("Running Camera Calibration")
      rsCam = RealSenseCamera(640,360)
      rsCamCalib = CalibrateCamera(rsCam)
      rsCamToRobot =   rsCamCalib.getCamera2Robot(tag_size=cfg.SIZE_OF_CALIBRATION_TAG, tag_id=0,viz=True)
      print("rsCamToRobot = \n", rsCamToRobot)

      # daiCam = DepthAICamera(1280,720, object_detection=False)
      daiCam = DepthAICamera(1920,1080,object_detection=False)
      daiCamCalib = CalibrateCamera(daiCam)
      # daiCam2LM = daiCamCalib.calibrateCameraWrtLandMark(tag_size=SIZE_OF_CALIBRATION_TAG, tag_id=0, viz=True)
      # print("daiCam2LM = \n", daiCam2LM)
      daiCamToRobot =   daiCamCalib.getCamera2Robot(tag_size=cfg.SIZE_OF_CALIBRATION_TAG, tag_id=0, viz=True)
      print("daiCamToRobot = \n", daiCamToRobot)      #save both the calibration matrices to a file "calib.txt" for later loading into np array
      np.savetxt("calib.txt", np.concatenate((rsCamToRobot, daiCamToRobot), axis=0), delimiter=",")     
  elif input == "Load":
      # load the camera calibration matrices from the file "calib.txt"
      print("loading th camera calibration from file")
      calib = np.loadtxt("calib.txt", delimiter=",")
      rsCamToRobot = calib[:4,:]
      daiCamToRobot = calib[4:,:]
      print("rsCamToRobot = \n", rsCamToRobot)
      print("daiCamToRobot = \n", daiCamToRobot)
  elif input == "calib_setup2":
      print("Running Camera Calibration")
      rsCam = RealSenseCamera(1280, 720)
      rsCamCalib = CalibrateCamera(rsCam)
      rsCamToRobot =   rsCamCalib.getCamera2Robot(tag_size=cfg.SIZE_OF_CALIBRATION_TAG, tag_id=0,viz=True)
      print("rsCamToRobot = \n", rsCamToRobot)


      input = input("Press c  to continue calibrating dai camera or q to quit")
      if input == "q":
        np.savetxt("calib.txt", np.concatenate((rsCamToRobot, np.loadtxt("calib.txt", delimiter=",")[4:,:]), axis=0), delimiter=",")     
        exit(0)
      elif input == "c":
        print("continuing")
        # daiCam = DepthAICamera(1280,720, object_detection=False)
        daiCam = DepthAICamera(1920,1080,object_detection=False)
        daiCamCalib = CalibrateCamera(daiCam)
        # daiCam2LM = daiCamCalib.calibrateCameraWrtLandMark(tag_size=SIZE_OF_CALIBRATION_TAG, tag_id=0, viz=True)
        # print("daiCam2LM = \n", daiCam2LM)
        daiCamToRobot =   daiCamCalib.getCamera2Robot(tag_size=cfg.SIZE_OF_CALIBRATION_TAG, tag_id=0, viz=True)
        print("daiCamToRobot = \n", daiCamToRobot)      #save both the calibration matrices to a file "calib.txt" for later loading into np array
        np.savetxt("calib.txt", np.concatenate((rsCamToRobot, daiCamToRobot), axis=0), delimiter=",")     
  else:
      print("incorrect argument")
  
  return rsCamToRobot, daiCamToRobot

import time
from multiprocessing import Process  

def daiTagPose():
  daiCam = DepthAICamera(width=1280, height=720, object_detection=False, viz=True)
  while True:
    daiCam.getTagPoses()

def daiObjectDetection():
  # daiCam = DepthAICamera(1280,720, object_detection=True, viz=True)
  daiCam = DepthAICamera(width=1280, height=720, object_detection=True, viz=True)
  while True:
    daiCam.runObjectDetection()

def goToGoalPosition():
   robot = VexCortex("/dev/ttyUSB0")
   while True:
       if robot.running():
        # get the timestamp, x,y,z position, and confidence score from the queue
        # if the queue is empty, it will throw an exception
        try:
            timestamp, x, y, z, confidence = daiQueue.get()
        except queue.Empty:
            print("queue is empty")
            continue
        #convert x,y,z to cm 
        x = x/10
        y = y/10
        z = z/10
        print("received value from queue")
        print("timestamp = ", timestamp)
        print("y = ", y)
        time.sleep(1)
        # if y > 10:
        #   drive_forward(robot,30)
        #   stop_drive(robot)
        # else:
        #   drive_backward(robot,30)
        #   stop_drive(robot)
       else:
        print("robot is not running")
        time.sleep(1)
      
import landmarks as lm
if __name__ == "__main__":
  np.set_printoptions(precision=2, suppress=True)
  if len(sys.argv) > 1:
    if sys.argv[1] == "calib_setup1":
      runCameraCalib("calib_setup1")
      exit(0)
    elif sys.argv[1] == "calib_setup2":
      runCameraCalib("calib_setup2")
      exit(0)
    elif sys.argv[1] == "calib":
      runCameraCalib("calib_setup1")
      exit(0)
    else:
      print("incorrect argument")
      exit(0)

  # for tagid, pose in lm.map_apriltag_poses_home:
  #   print("tagid = ", tagid)
  #   print("pose = ", pose)

  #create a queue to store the timestamp, x,y,z position, and confidence score, and pass it to the camera object 
  daiObjectDetection()
  # daiTagPose()
  # create multiple process, one for object detection and one for robot control
  # p1 = Process(target=daiObjectDetection, args=())
  # p1.start()
  # # time.sleep(10)
  # p2 = Process(target=goToGoalPosition, args=())
  # p2.start()
  # p1.join()
  # p2.join()
  # daiCam.runObjectDetection()
  # rsCamToRobot, daiCamToRobot = runCameraCalib("Load")
  # print("rsCamToRobot = \n", rsCamToRobot)


  # daiCam2Robot = daiCam2LM @ np.linalg.inv(rsCam2LM) @ rsCam2Robot

  # robot = VexCortex("/dev/ttyUSB0")
  # testTransulateAlongY(robot, cam)

# testRotate(robot)
# robot.motor[0] = 0 #if theta > 0 else -127
# robot.motor[9] = 0 #if theta > 0 else -127

# rsCamToRobot = 
#  [[ 1.   -0.03 -0.07 12.89]
#  [-0.08 -0.25 -0.96 15.03]
#  [ 0.01  0.97 -0.26 16.34]
#  [ 0.    0.    0.    1.  ]]

# daiCamToRobot = 
#  [[  1.    -0.04   0.04 -12.07]
#  [  0.03  -0.2   -0.98  15.03]
#  [  0.05   0.98  -0.2   11.9 ]
#  [  0.     0.     0.     1.  ]]
