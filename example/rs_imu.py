import pyrealsense2 as rs
import math
import time

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
align = rs.align(rs.stream.color)


# Start the pipeline
pipeline.start(config)

'''Use the accelerometer and gyroscope data to calculate the pose of the camera'''
# Let's assume the camera is at rest on a flat surface. The acceleration due to gravity is 9.8 m/s^2
# The camera is at an angle to the ground, so the acceleration due to gravity is not 9.8 m/s^2.
# Use the acceleration data to calculate the angle of the camera.
# https://www.nxp.com/docs/en/application-note/AN3461.pdf
# https://www.nxp.com/docs/en/application-note/AN3463.pdf
# Calculate the angle of the camera in degrees
angle = math.atan2(accel_data.y, accel_data.z) * 180 / math.pi
print(f"Angle: {angle}")



    



    

        
try:
    prev_time = time.time()
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        # Get accelerometer and gyroscope data and add it to the frameset
        for frame in frames:
            if frame.is_motion_frame():
                motion_data = frame.as_motion_frame()
                if motion_data.get_profile().stream_type() == rs.stream.accel:
                    accel_data = motion_data.get_motion_data()
                    print(f"Accelerometer Data (m/s^2): X={accel_data.x}, Y={accel_data.y}, Z={accel_data.z}")
                    # The d435i camera + IMU is at an angle to the ground, so the acceleration due to gravity is not 9.8 m/s^2. 
                    # Use the acceleration data to calculate the angle of the camera.
                    # https://www.nxp.com/docs/en/application-note/AN3461.pdf
                    # https://www.nxp.com/docs/en/application-note/AN3463.pdf
                    # Calculate the angle of the camera in degrees
                    angle = math.atan2(accel_data.y, accel_data.z) * 180 / math.pi
                    print(f"Angle: {angle}")    
                elif motion_data.get_profile().stream_type() == rs.stream.gyro:
                    gyro_data = motion_data.get_motion_data()
                    print(f"Gyroscope Data (rad/s): X={gyro_data.x}, Y={gyro_data.y}, Z={gyro_data.z}")
        # Caluculate the time difference between frames and print fps
        dt = time.time() - prev_time
        print(f"FPS: {1/dt}")
        prev_time = time.time()



finally:
    pipeline.stop()
