# this python module does a few things
# 1. It initilizes the realsense d435i camera
# 2. It reads the aligned frames that contain depth, color, accel, and gyro data image from the camera
# 3. It converts the realsense frames to open3d format
# 4. It uses the open3d rgbd image to create a point cloud
# 5. It uses the open3d point cloud to create a mesh
# 6. Use the camera, accel and gyro to calculate the camera pose
# 7. Use the camera pose to create a trajectory
# 8. Visualize the trajectory and the mesh
print("The script doesn't work yet")
exit(0)

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import math
from scipy.spatial.transform import Rotation as R
import time

# 1. It initilizes the realsense d435i camera
def inilitizeCamera():
    try:
      # Create a context object
      ctx = rs.context()
      # Get a device
      dev = ctx.devices[0]
      # Reset the camera
      dev.hardware_reset()
    except:
      print("No realsense camera found, or failed to reset.")
    
    #initilize realsense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)
    return pipeline, align

# 2. It reads the aligned frames that contain depth, color, accel, and gyro data image from the camera
def get_realsense_frames(pipeline, align):
    #get realsense frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    return aligned_frames

# get depth frame
def get_depth_frame(frames):
    #get depth frame
    depth_frame = frames.get_depth_frame()
    return depth_frame

# get color frame
def get_color_frame(frames):
    #get color frame
    color_frame = frames.get_color_frame()
    return color_frame

# get accel frame
def get_accel_frame(frames):
    #get accel frame
    accel_frame = frames.first_or_default(rs.stream.accel)
    return accel_frame

# get gyro frame
def get_gyro_frame(frames):
    #get gyro frame
    gyro_frame = frames.first_or_default(rs.stream.gyro)
    return gyro_frame


# 3. It converts the realsense frames to open3d format
def convert_depth_and_color_image_to_open3d_format(depth_frame, color_frame):
    #convert realsense depth and color image to open3d rgbd image
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color_image), o3d.geometry.Image(depth_image), convert_rgb_to_intensity = False)
    return rgbd_image

#4. It uses the open3d rgbd image to create a point cloud
def convert_to_open3d_format(rgbd_image):
    #convert realsense point cloud to open3d point cloud
    o3d_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    return o3d_pcd

#5. It uses the open3d point cloud to create a mesh
def create_mesh(o3d_pcd):
    #create mesh
    o3d_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    distances = o3d_pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(o3d_pcd, o3d.utility.DoubleVector([radius, radius * 2]))
    return bpa_mesh

#6. Use the camera, accel and gyro to calculate the camera pose
def get_realsense_pose(frames):
    #get realsense pose
    pose = frames.get_pose_frame()
    if pose:
        pose = pose.get_pose_data()
        return pose
    else:
        return None
# Use visual inertial odometry to calculate the camera pose
def get_pose(frames):
    depth_frame = get_depth_frame(frames)
    color_frame = get_color_frame(frames)
    rgbd_image = convert_depth_and_color_image_to_open3d_format(depth_frame, color_frame)
    o3d_pcd = convert_to_open3d_format(rgbd_image)
    bpa_mesh = create_mesh(o3d_pcd)
    pose = get_realsense_pose(frames)
    pose_matrix = convert_pose_to_open3d_format(pose)
    return pose_matrix


    

#7. Use the camera pose to create a trajectory
def convert_pose_to_open3d_format(pose):
    #convert realsense pose to open3d pose
    pose_matrix = np.array(pose.get_rotation_matrix()).reshape(3, 3)
    pose_matrix = np.column_stack((pose_matrix, pose.get_translation()))
    pose_matrix = np.row_stack((pose_matrix, [0, 0, 0, 1]))
    return pose_matrix


#8. Visualize the trajectory and the mesh
def visualize(trajectory, bpa_mesh):
    #visualize trajectory and mesh
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(bpa_mesh)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_matrix(trajectory)
    vis.run()
    vis.destroy_window()

#use the accel and gyro to calculate the camera pose, and then use the camera pose to create a trajectory
def get_pose(accel_frame, gyro_frame, pose):
    #get accel and gyro data
    accel_data = accel_frame.as_motion_frame().get_motion_data()
    gyro_data = gyro_frame.as_motion_frame().get_motion_data()
    #get accel and gyro timestamp
    accel_ts = accel_frame.get_timestamp()
    gyro_ts = gyro_frame.get_timestamp()
    #get pose timestamp
    pose_ts = pose.timestamp
    #calculate the time difference between the pose and accel/gyro
    dt1 = (accel_ts - pose_ts) * 1e-3
    dt2 = (gyro_ts - pose_ts) * 1e-3
    #calculate the accel and gyro data
    accel = np.array([accel_data.x, accel_data.y, accel_data.z])
    gyro = np.array([gyro_data.x, gyro_data.y, gyro_data.z])
    #calculate the rotation matrix
    R = pose.rotation_matrix
    #calculate the translation vector
    t = pose.translation
    #calculate the rotation matrix derivative
    R_dot = np.array([[-gyro[0], -gyro[1], -gyro[2]],
                      [gyro[0], 0, -gyro[2]],
                      [gyro[1], gyro[2], 0]])
    #calculate the translation vector derivative
    t_dot = accel
    #calculate the pose derivative
    pose_dot = np.zeros((4, 4))
    pose_dot[0:3, 0:3] = R_dot
    pose_dot[0:3, 3] = t_dot
    #calculate the pose
    pose = np.dot(pose, pose_dot * dt1)
    #calculate the rotation matrix
    R = pose[0:3, 0:3]
    #calculate the translation vector
    t = pose[0:3, 3]
    #calculate the rotation vector
    r = R.from_matrix(R).as_rotvec()
    #calculate the rotation vector derivative
    r_dot = np.dot(R_dot, r)
    #calculate the rotation matrix derivative
    R_dot = R.from_rotvec(r_dot).as_matrix()
    #calculate the pose derivative
    pose_dot = np.zeros((4, 4))
    pose_dot[0:3, 0:3] = R_dot
    pose_dot[0:3, 3] = t_dot


# main function
def main():
    #initilize realsense camera
    pipeline, align = inilitizeCamera()

    #get realsense frames
    frames = get_realsense_frames(pipeline, align)

    #get depth frame
    depth_frame = get_depth_frame(frames)

    #get color frame
    color_frame = get_color_frame(frames)

    #get accel frame
    # accel_frame = get_accel_frame(frames)

    #get gyro frame
    # gyro_frame = get_gyro_frame(frames)

    #convert realsense depth and color image to open3d rgbd image
    rgbd_image = convert_depth_and_color_image_to_open3d_format(depth_frame, color_frame)

    #convert realsense point cloud to open3d point cloud
    o3d_pcd = convert_to_open3d_format(rgbd_image)

    #create mesh
    bpa_mesh = create_mesh(o3d_pcd)

    #get realsense pose
    pose = get_realsense_pose(frames)

    #convert realsense pose to open3d pose
    pose_matrix = convert_pose_to_open3d_format(pose)

    #visualize trajectory and mesh
    visualize(pose_matrix, bpa_mesh)

if __name__ == "__main__":
    main()

