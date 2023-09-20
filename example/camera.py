import numpy as np
import pyrealsense2 as rs
from typing import Tuple
import cv2
from datetime import datetime
from pyapriltags import Detector
from scipy.spatial.transform import Rotation as R



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

  def getCameraParams(self):
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
      cv2.destroyAllWindows()

  def read(self, scale=False): # will also store in buffer to be read
    ret, color, depth = self.capture()
    if not ret:
      return np.zeros((self.height, self.width, 3), dtype=np.uint8), \
             np.zeros((self.height, self.width), dtype=np.float32)
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

  def getTagPose(self, tag_size, input_type="color"): # other input type is filtered_color
    at_detector = Detector(families='tag16h5',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=1,
                           debug=0)
    camera_params = self.getCameraParams()
    # detect the tag and get the pose
    while True:
      if input_type == "color":
        color, depth = self.read()
        self.filtered_color = color    # TODO vairable cleanup 
      elif input_type == "filtered_color":
        self.getFilteredColorBasedOnDepth()
       
      tags = at_detector.detect(
      cv2.cvtColor(self.filtered_color, cv2.COLOR_BGR2GRAY), True, camera_params[0:4], tag_size)
      found_tag = False
      for tag in tags:
        if tag.decision_margin < 50: 
          continue
        found_tag = True


      if self.debug_mode:
        if not found_tag:
          cv2.imshow("color", self.filtered_color)
          if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            exit(0)
          continue
        for tag in tags:
            if tag.decision_margin < 70:
                continue
            # print the tag pose, tag id, etc well formatted
            # print("Tag Family: ", tag.tag_family)
            # print("Tag ID: ", tag.tag_id)
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
  
# if the file is run directly
if __name__ == "__main__":
  cam = RealSenseCamera(1280, 720, True)
  while True:
    # cam.view()
    # cam.getFilteredColorBasedOnDepth()
    cam.getTagPose(tag_size=7)
    