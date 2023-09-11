import pyrealsense2 as rs
import numpy as np
from datetime import datetime


def initialize_camera():
    p = rs.pipeline()
    conf = rs.config()
    conf.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)
    conf.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200) 
    prof = p.start(conf)
    return p
    

def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])


def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])

p = initialize_camera()

try:
    dt = datetime.now()
    sensor_values = []
    sample_count = 1000
    for _ in range(sample_count):
        f = p.wait_for_frames()
        accel = accel_data(f[0].as_motion_frame().get_motion_data())
        gyro = gyro_data(f[1].as_motion_frame().get_motion_data())
        sensor_values.append(np.concatenate((accel, gyro)))
    process_time = (datetime.now() - dt).total_seconds() 
    process_time_per_sample = process_time / sample_count
    fps = 1 / process_time_per_sample
    # print process time and fps
    print("process time: ", process_time)
    print("process time per sample: ", process_time_per_sample)
    print("fps: ", fps)
    print("accelerometer: ", accel)
    print("gyro: ", gyro)

finally:
    p.stop() 