import pyrealsense2 as rs
import numpy as np
import cv2
import os
import csv
import datetime

# Function to create a directory if it doesn't exist
def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

# Set up directories and files for saving data
color_dir = "color_data"
depth_dir = "depth_data"
accel_file = "accel_data.csv"
gyro_file = "gyro_data.csv"

ensure_dir(color_dir)
ensure_dir(depth_dir)

# Configure streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

# Configure accel data to the highest frame rate
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)  # Example rate
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)   # Example rate

pipeline.start(config)

# Open CSV files for accel data
with open(accel_file, mode='w', newline='') as accel_csv, open(gyro_file, mode='w', newline='') as gyro_csv:
    accel_writer = csv.writer(accel_csv)
    gyro_writer = csv.writer(gyro_csv)

    try:
        while True:
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            accel_frame = frames.first(rs.stream.accel)
            gyro_frame = frames.first(rs.stream.gyro)

            if not depth_frame or not color_frame or not accel_frame or not gyro_frame:
                continue

            # Process and save data
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")

            # Save color and depth frames
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            color_filename = f"{color_dir}/color_{timestamp}.png"
            depth_filename = f"{depth_dir}/depth_{timestamp}.png"
            cv2.imwrite(color_filename, color_image)
            cv2.imwrite(depth_filename, depth_image)

            # Save accel data
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            accel_writer.writerow([timestamp, accel_data.x, accel_data.y, accel_data.z])
            gyro_writer.writerow([timestamp, gyro_data.x, gyro_data.y, gyro_data.z])

            # Display images for testing
            cv2.imshow('Depth Image', depth_image)
            cv2.imshow('Color Image', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
