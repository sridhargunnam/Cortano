import open3d
import camera
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyapriltags import Detector

if __name__ == "__main__":
  # robot = RemoteInterface("...")
  cam = camera.RealsenseCamera()

  prev_rgbd_image = None
  option = open3d.pipelines.odometry.OdometryOption()
  # Intel Realsense D415 Intrinsic Parameters
  fx = 460.92495728   # FOV(x) -> depth2xyz -> focal length (x)
  fy = 460.85058594   # FOV(y) -> depth2xyz -> focal length (y)
  cx = 315.10949707   # 640 (width) 320
  cy = 176.72598267   # 360 (height) 180
  cam_intrinsic_params = open3d.camera.PinholeCameraIntrinsic(640, 360, fx, fy, cx, cy)
  camera_params = ( fx, fy, cx, cy )
  at_detector = Detector(families='tag16h5',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

  global_T = np.identity(4)
  # while robot.running():
  while True:
    # color, depth, sensor = robot.read()
    color, depth = cam.read()

    # filter the depth image so that noise is removed
    depth = depth.astype(np.float32) / 1000.
    mask = np.bitwise_and(depth > 0.1, depth < 3.0) # -> valid depth points
    filtered_depth = np.where(mask, depth, 0)

    # converting to Open3D's format so we can do odometry
    o3d_color = open3d.geometry.Image(color)
    o3d_depth = open3d.geometry.Image(filtered_depth)
    o3d_rgbd  = open3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth)

    tags = at_detector.detect(
      cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), True, camera_params, 2.5)
    found_tag = False
    for tag in tags:
      if tag.decision_margin < 50: continue
      #found_tag!
      if tag.tag_id == 1: # 1..8
        print(tag.pose_R, tag.pose_t)
        # global_T = get_pose(tag.pose_R, tag.pose_t) # use your get_pose algorithm here!
        found_tag = True
      
    if not found_tag and prev_rgbd_image is not None: # use RGBD odometry relative transform to estimate pose
      T = np.identity(4)
      ret, T, _ = open3d.pipelines.odometry.compute_rgbd_odometry(
        o3d_rgbd, prev_rgbd_image, cam_intrinsic_params, T,
        open3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
      global_T = global_T.dot(T)
      rotation = global_T[:3,:3]
      print("Rotation: ", R.from_matrix(rotation).as_rotvec(degrees=True))

    prev_rgbd_image = o3d_rgbd # we forgot this last time!

    # dont need this, but helpful to visualize
    filtered_color = np.where(np.tile(mask.reshape(360, 640, 1), (1, 1, 3)), color, 0)
    cv2.imshow("color", filtered_color)
    cv2.waitKey(1)