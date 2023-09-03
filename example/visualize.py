# visualizing camera trajectory in the 3d point cloud in open3d. 
# For this use the realsense depth camera d435i to get the depth, color and imu samples. 
# Finally show the visualization of the camera trajectory. 
# The camera trajectory can be obtained by using the realsense pose sensor.

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time
from scipy.spatial.transform import Rotation as R   

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.pose)
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)
# Start streaming
pipeline.start(config)

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)

# Create a 3D point cloud using the camera feed. 
# The point cloud is created using the depth and color data from the camera.
pc = rs.pointcloud()
# We want the points object to be persistent so we can display the last cloud when a frame drops
points = rs.points()

#visualize the camera trajectory of real sense in the 3d point cloud in open3d
def visualize_camera_trajectory_in_3d_point_cloud_in_open3d():
    #visualize the camera trajectory of real sense in the 3d point cloud in open3d
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = get_realsense_frames(pipeline, align)
            pose = get_realsense_pose(frames)
            if pose:
                #convert realsense pose to open3d pose
                pose_matrix = convert_pose_to_open3d_format(pose)
                #convert realsense depth and color image to open3d rgbd image
                rgbd_image = convert_depth_and_color_image_to_open3d_format(frames.get_depth_frame(), frames.get_color_frame())
                #convert realsense point cloud to open3d point cloud
                o3d_pcd = convert_to_open3d_format(rgbd_image)
                #visualize the camera trajectory in the 3d point cloud in open3d
                custom_draw_geometry_with_camera_trajectory(o3d_pcd)
    finally:
        pipeline.stop()


#visualize the camera trajectory in the 3d point cloud in open3d
def custom_draw_geometry_with_camera_trajectory(pcd):
    #visualize the camera trajectory in the 3d point cloud in open3d
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

#convert realsense depth image to open3d depth image
def convert_depth_image_to_open3d_format(depth_image):
    #convert realsense depth image to open3d depth image
    depth_image = np.asanyarray(depth_image)
    depth_image = cv2.flip(depth_image, 1)
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
    depth_image = o3d.geometry.Image(depth_image)
    return depth_image

#convert realsense color image to open3d color image
def convert_color_image_to_open3d_format(color_image):
    #convert realsense color image to open3d color image
    color_image = np.asanyarray(color_image)
    color_image = cv2.flip(color_image, 1)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    color_image = o3d.geometry.Image(color_image)
    return color_image

#convert realsense depth and color image to open3d rgbd image
def convert_depth_and_color_image_to_open3d_format(depth_image, color_image):
    #convert realsense depth and color image to open3d rgbd image
    depth_image = np.asanyarray(depth_image)
    depth_image = cv2.flip(depth_image, 1)
    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
    depth_image = o3d.geometry.Image(depth_image)
    color_image = np.asanyarray(color_image)
    color_image = cv2.flip(color_image, 1)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    color_image = o3d.geometry.Image(color_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, depth_scale=1.0, depth_trunc=100.0, convert_rgb_to_intensity=False)
    return rgbd_image



#convert realsense point cloud to open3d point cloud
def convert_to_open3d_format(rgbd_image):
    #convert realsense point cloud to open3d point cloud
    o3d_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    return o3d_pcd

#convert realsense pose to open3d pose
def convert_pose_to_open3d_format(pose):
    #convert realsense pose to open3d pose
    pose_matrix = np.array(pose.get_rotation_matrix()).reshape(3, 3)
    pose_matrix = np.column_stack((pose_matrix, pose.get_translation()))
    pose_matrix = np.row_stack((pose_matrix, [0, 0, 0, 1]))
    return pose_matrix



def get_realsense_pose(frames):
    #get realsense pose
    pose = frames.get_pose_frame()
    if pose:
        pose = pose.get_pose_data()
        return pose
    else:
        return None

#get realsense frames that contain depth, color and pose data
def get_realsense_frames(pipeline, align):
    #get realsense frames that contain depth, color and pose data
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    return aligned_frames

    
# Finally show the visualization of the camera trajectory. 
# The camera trajectory can be obtained by using the realsense pose sensor.
# Create a visualization window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Set up the robot arm model (you can load your own 3D model here)
robot_arm_model = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
vis.add_geometry(robot_arm_model)

# Simulation loop
for angle in range(0, 360, 5):
    # Compute the robot arm's new position (replace with your own logic)
    x = np.cos(np.radians(angle))
    y = np.sin(np.radians(angle))
    z = 0.5  # Fixed height

    # Update the trajectory PointCloud
    robot_arm_trajectory.points.append([x, y, z])

    # Update the visualization
    vis.update_geometry(robot_arm_trajectory)
    vis.poll_events()
    vis.update_renderer()

    # Pause briefly to control the animation speed
    time.sleep(0.05)
    























# import open3d as o3d
# import numpy as np

# # Load camera poses and 3D point cloud (replace with your data loading code)
# camera_poses = []  # List of camera poses as 4x4 transformation matrices
# point_cloud = o3d.geometry.PointCloud()  # Your 3D point cloud data

# # Create a visualization window
# vis = o3d.visualization.Visualizer()
# vis.create_window()

# # Add the 3D point cloud to the visualization
# vis.add_geometry(point_cloud)

# # Visualization loop
# for pose in camera_poses:
#     # Create a camera model at the current pose (replace with your camera model)
#     camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
#     camera.transform(pose)  # Apply the camera pose

#     # Add the camera model to the visualization
#     vis.add_geometry(camera)

#     # Update the visualization
#     vis.poll_events()
#     vis.update_renderer()

# # Keep the window open until you manually close it
# vis.run()
# vis.destroy_window()




# import open3d as o3d
# import numpy as np
# import time

# # Create an empty PointCloud to represent the robot arm's trajectory
# robot_arm_trajectory = o3d.geometry.PointCloud()

# # Create a visualization window
# vis = o3d.visualization.Visualizer()
# vis.create_window()

# # Set up the robot arm model (you can load your own 3D model here)
# robot_arm_model = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
# vis.add_geometry(robot_arm_model)

# # Simulation loop
# for angle in range(0, 360, 5):
#     # Compute the robot arm's new position (replace with your own logic)
#     x = np.cos(np.radians(angle))
#     y = np.sin(np.radians(angle))
#     z = 0.5  # Fixed height

#     # Update the trajectory PointCloud
#     robot_arm_trajectory.points.append([x, y, z])

#     # Update the visualization
#     vis.update_geometry(robot_arm_trajectory)
#     vis.poll_events()
#     vis.update_renderer()

#     # Pause briefly to control the animation speed
#     time.sleep(0.05)

# # Close the visualization window
# vis.destroy_window()
