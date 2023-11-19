import pyrealsense2 as rs
import numpy as np
import cv2
import time
import math

def initialize_camera():
    # Configure depth and IMU streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)
    pipeline.start(config)
    return pipeline

def process_imu_data(pipeline):
    # Variables to store accumulated values
    accel_data = np.zeros(3)
    gyro_data = np.zeros(3)
    last_reset_time = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()

            # Get IMU data
            accel = frames[0].as_motion_frame().get_motion_data()
            gyro = frames[1].as_motion_frame().get_motion_data()

            # Update accumulated values
            accel_data += np.array([accel.x, accel.y, accel.z])
            gyro_data += np.array([gyro.x, gyro.y, gyro.z])

            # Compute orientation from gyro data
            current_time = time.time()
            if current_time - last_reset_time > 5:
                # Reset the pose
                accel_data[:] = 0
                gyro_data[:] = 0
                last_reset_time = current_time

            # Visualization
            visualize_orientation(gyro_data)

    finally:
        pipeline.stop()

def visualize_orientation(gyro_data):
    # Create a blank image
    img = np.zeros((500, 500, 3), dtype=np.uint8)

    center = (250, 250)
    scale = 100
    # Calculate the arrow direction
    angle = math.atan2(gyro_data[1], gyro_data[0])
    end_point = (int(250 + 100 * math.cos(angle)), int(250 + 100 * math.sin(angle)))

    # Draw the arrow
    cv2.arrowedLine(img, (250, 250), end_point, (0, 255, 0), 5)

    # Show the image
    cv2.imshow('Orientation', img)
    cv2.waitKey(1)

# if __name__ == "__main__":
#     pipeline = initialize_camera()
#     process_imu_data(pipeline)

import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def process_gyro(gyro_data, delta_t):
    # Implement your gyro processing algorithm here
    # For example, a simple integration to estimate angular position
    return gyro_data * delta_t

def process_accel(accel_data):
    # Implement your accelerometer processing algorithm here
    return accel_data

def draw_arrow(ax, position, orientation, color):
    ax.quiver(position[0], position[1], position[2], 
              orientation[0], orientation[1], orientation[2], 
              color=color)

def main():
    # Configure streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)
    pipeline.start(config)

    try:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        while True:
            frames = pipeline.wait_for_frames()

            accel_frame = frames.first(rs.stream.accel).as_motion_frame()
            gyro_frame = frames.first(rs.stream.gyro).as_motion_frame()
            if accel_frame and gyro_frame:
                accel_data = np.array([accel_frame.get_motion_data().x, 
                                    accel_frame.get_motion_data().y, 
                                    accel_frame.get_motion_data().z])
                gyro_data = np.array([gyro_frame.get_motion_data().x, 
                                    gyro_frame.get_motion_data().y, 
                                    gyro_frame.get_motion_data().z])

                delta_t = gyro_frame.get_timestamp() - accel_frame.get_timestamp()

                gyro_orientation = process_gyro(gyro_data, delta_t)
                accel_orientation = process_accel(accel_data)

                ax.clear()
                draw_arrow(ax, [0, 0, 0], gyro_orientation, 'r')
                draw_arrow(ax, [1, 1, 1], accel_orientation, 'b')

                plt.pause(0.1)

    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
