print("the script doesn't work yet")
exit(0)
import pyrealsense2 as rs
import numpy as np
import cupoch as cph
from cortano import RemoteInterface
import numpy as np
from datetime import datetime
import multiprocessing
import queue

# Initialize the cuPOC device
cph.initialize_allocator(cph.PoolAllocation, 1000000000)

# depth
fx = 460.92495728   # FOV(x) -> depth2xyz -> focal length (x)
fy = 460.85058594   # FOV(y) -> depth2xyz -> focal length (y)
cx = 315.10949707   # 640 (width) 320
cy = 176.72598267   # 360 (height) 180

# Create a cuPOC pinhole camera intrinsic
intrinsic = cph.camera.PinholeCameraIntrinsic(
    width=640,
    height=480,
    fx=fx,
    fy=fy,
    cx=cx,
    cy=cy,
)

# Initialize the cuPOC RGBD camera
rgbd = cph.camera.RgbdCamera(intrinsic)

# Initialize the cuPOC RGBD Odometry
odometry = cph.odometry.RgbdOdometryJacobianFromHybridTerm()

# Create a visualization window
vis = cph.visualization.Visualizer()

# Create a 3D map
pcd = cph.geometry.PointCloud()

# Create a pipeline for reading frames from the .bag file
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file("/home/sgunnam/Documents/20230906_080949.bag")

# Start the pipeline
pipeline.start(config)
align = rs.align(rs.stream.color)

try:
    while True:
        # Read a frame from the .bag file
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_frame = np.asanyarray(depth_frame.get_data())
        color_frame = np.asanyarray(color_frame.get_data())
        if not depth_frame or not color_frame:
            continue

        # Convert depth and color frames to cuPOC images
        depth_frame = depth_frame.astype(np.float32)
        # depth_image = cph.geometry.Image(np.array(depth_frame.get_data()))
        # color_image = cph.geometry.Image(np.array(color_frame.get_data()))
        color_image = cph.geometry.Image(color_frame)
        depth_image = cph.geometry.Image(depth_frame)
        

        
        # Integrate the frame into the 3D map
        rgbd.integrate(depth_image, color_image, odometry)

        # Update the visualization
        vis.add_geometry(pcd)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()

except KeyboardInterrupt:
    pass

finally:
    pipeline.stop()
    vis.destroy_window()
