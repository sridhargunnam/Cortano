
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import json
import pyrealsense2 as rs
import time

class Vector3D:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_measurement = 0
        self.error_covariance = 1

    def update(self, measurement):
        kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_variance)
        self.estimated_measurement += kalman_gain * (measurement - self.estimated_measurement)
        self.error_covariance = (1 - kalman_gain) * self.error_covariance + self.process_variance
        return self.estimated_measurement

def apply_calibration(imu_data, calibration):
    imu_data.x -= calibration["bias"][0]
    imu_data.y -= calibration["bias"][1]
    imu_data.z -= calibration["bias"][2]
    scale_and_alignment = np.array(calibration["scale_and_alignment"]).reshape(3, 3)
    adjusted_data = scale_and_alignment @ np.array([imu_data.x, imu_data.y, imu_data.z])
    return Vector3D(*adjusted_data)

def process_gyro(gyro_data, kalman_filter):
    gyro_data.x = kalman_filter.update(gyro_data.x)
    gyro_data.y = kalman_filter.update(gyro_data.y)
    gyro_data.z = kalman_filter.update(gyro_data.z)
    return gyro_data

def process_accel(accel_data, kalman_filter):
    accel_data.x = kalman_filter.update(accel_data.x)
    accel_data.y = kalman_filter.update(accel_data.y)
    accel_data.z = kalman_filter.update(accel_data.z)
    return accel_data

# Load calibration data
with open('/home/nvidia/wsp/clawbot/Cortano/imu_calib/calibration.json', 'r') as file:
    calibration_data = json.load(file)

accel_calibration = calibration_data["imus"][0]["accelerometer"]
gyro_calibration = calibration_data["imus"][0]["gyroscope"]

# RealSense pipeline setup for IMU data
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)

# Start the pipeline
pipeline.start(config)

# Initialize Kalman filters
gyro_kalman_filter = KalmanFilter(process_variance=0.1, measurement_variance=1)
accel_kalman_filter = KalmanFilter(process_variance=0.1, measurement_variance=1)

# Set up the live plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

quiver_gyro = ax.quiver(0, 0, 0, 0, 0, 0, color='r')
quiver_accel = ax.quiver(0, 0, 0, 0, 0, 0, color='b')

def update_plot(frame):
    frames = pipeline.wait_for_frames()
    gyro_frame = frames.first_or_default(rs.stream.gyro)
    accel_frame = frames.first_or_default(rs.stream.accel)

    if gyro_frame and accel_frame:
        gyro_data = gyro_frame.as_motion_frame().get_motion_data()
        accel_data = accel_frame.as_motion_frame().get_motion_data()

        gyro_vector = Vector3D(gyro_data.x, gyro_data.y, gyro_data.z)
        accel_vector = Vector3D(accel_data.x, accel_data.y, accel_data.z)

        calibrated_gyro = apply_calibration(gyro_vector, gyro_calibration)
        calibrated_accel = apply_calibration(accel_vector, accel_calibration)

        processed_gyro = process_gyro(calibrated_gyro, gyro_kalman_filter)
        processed_accel = process_accel(calibrated_accel, accel_kalman_filter)

        quiver_gyro.set_segments([[[0, 0, 0], [processed_gyro.x, processed_gyro.y, processed_gyro.z]]])
        quiver_accel.set_segments([[[0, 0, 0], [processed_accel.x, processed_accel.y, processed_accel.z]]])

ani = FuncAnimation(fig, update_plot, blit=False, interval=100)
plt.show()

try:
    pass  # The animation loop is handled by FuncAnimation
finally:
    # Stop the pipeline
    pipeline.stop()



from pyapriltags import Detector
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np

def detect_april_tag(image):
    # Initialize AprilTag detector
    detector = Detector()
    # Detect tags
    tags = detector.detect(image)
    return tags

def estimate_tag_pose(tag):
    # Estimate the pose of the detected tag (simplified)
    # Actual implementation would depend on the specific requirements and camera calibration
    # For demonstration, returning a mock pose
    return np.array([0, 0, 0]), R.from_euler('xyz', [0, 0, 0]).as_quat()


# Modify the update_plot function to include AprilTag detection and pose estimation
def update_plot_with_april_tag(frame):
    # Existing code to get IMU data
    # ...

    # Capture image from the RealSense camera for AprilTag detection
    # This part needs the camera streaming setup which is currently not shown in the script
    # For demonstration, assuming an image is captured and stored in 'image'
    image = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder for actual image capture

    # Detect AprilTags in the image
    tags = detect_april_tag(image)

    # If tags are detected, estimate their poses and update the Kalman filter
    if tags:
        for tag in tags:
            tag_position, tag_orientation = estimate_tag_pose(tag)
            # Update the Kalman filter with this accurate pose information
            # ...

    # Continue with existing IMU data processing and visualization
    # ...

# Modify the FuncAnimation call to use the new update function
ani = FuncAnimation(fig, update_plot_with_april_tag, blit=False, interval=100)

plt.show()
