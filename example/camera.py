import numpy as np
import pyrealsense2 as rs
from typing import Tuple
import cv2

'''
#    Translation Vector: 0.00552000012248755  -0.00510000018402934  -0.011739999987185  

# Extrinsic from "Accel"	  To	  "Color" :
#  Rotation Matrix:
#    0.99982          0.0182547        0.00521857    
#   -0.0182514        0.999833        -0.000682113   
#   -0.00523015       0.000586744      0.999986   
'''
class RealSenseCamera:
  def __init__(self, width=640, height=360):
    self.width = width
    self.height = height
    self.shape = (height, width)

    try:
      # Create a context object
      ctx = rs.context()
      # Get a device
      dev = ctx.devices[0]
      # Reset the camera
      dev.hardware_reset()
    except:
      print("No realsense camera found, or failed to reset.")
    
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

# if the file is run directly
if __name__ == "__main__":
  cam = RealSenseCamera()
  while True:
    cam.view()
    