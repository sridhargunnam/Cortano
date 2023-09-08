print("the scrpt does not work")
exit(0)
import pyrealsense2 as rs
import numpy as np
import cv2
import OpenGL.GL as gl
from OpenGL.GLUT import glutSolidSphere

def draw_axes():
    gl.glLineWidth(2)
    gl.glBegin(gl.GL_LINES)
    # Draw x, y, z axes
    gl.glColor3f(1, 0, 0); gl.glVertex3f(0, 0, 0);  gl.glVertex3f(-1, 0, 0)
    gl.glColor3f(0, 1, 0); gl.glVertex3f(0, 0, 0);  gl.glVertex3f(0, -1, 0)
    gl.glColor3f(0, 0, 1); gl.glVertex3f(0, 0, 0);  gl.glVertex3f(0, 0, 1)
    gl.glEnd()
    gl.glLineWidth(1)

def draw_floor():
    gl.glBegin(gl.GL_LINES)
    gl.glColor4f(0.4, 0.4, 0.4, 1.0)
    # Render "floor" grid
    for i in range(9):
        gl.glVertex3i(i - 4, 1, 0)
        gl.glVertex3i(i - 4, 1, 8)
        gl.glVertex3i(-4, 1, i)
        gl.glVertex3i(4, 1, i)
    gl.glEnd()

def render_scene(offset_y, pitch, yaw):
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    gl.glColor3f(1.0, 1.0, 1.0)

    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.gluPerspective(60.0, 4.0 / 3.0, 1, 40)

    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glMatrixMode(gl.GL_MODELVIEW)

    gl.glLoadIdentity()
    gl.gluLookAt(1, 0, 5, 1, 0, 0, 0, -1, 0)

    gl.glTranslatef(0, 0, +0.5 + offset_y * 0.05)
    gl.glRotated(pitch, -1, 0, 0)
    gl.glRotated(yaw, 0, 1, 0)
    draw_floor()

def main():
    # Create a pipeline and configure it for IMU streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f)

    # Start streaming
    pipeline.start(config)

    offset_y = 0.0
    pitch = 0.0
    yaw = 0.0

    while True:
        try:
            frames = pipeline.wait_for_frames()

            # Extract accelerometer and gyro data
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()

            # Process gyro data
            # Calculate change in angle based on gyro data
            if 'last_ts_gyro' not in locals():
                last_ts_gyro = gyro_frame.get_timestamp()
            else:
                ts = gyro_frame.get_timestamp()
                dt_gyro = (ts - last_ts_gyro) / 1000.0
                last_ts_gyro = ts
                gyro_angle = np.array([gyro_data.x, gyro_data.y, gyro_data.z]) * dt_gyro

                # Update theta (rotation) based on gyro data
                if 'theta' not in locals():
                    theta = gyro_angle
                else:
                    theta -= gyro_angle

            # Process accelerometer data
            # Calculate rotation angle from accelerometer data
            accel_angle = np.array([np.arctan2(accel_data.y, accel_data.z), np.arctan2(accel_data.x, np.sqrt(accel_data.y * accel_data.y + accel_data.z * accel_data.z))])

            # Initialize theta with accelerometer data on the first iteration
            if 'theta' not in locals():
                theta = accel_angle
                theta[1] = np.pi  # Set Y-axis initial pose

            # Apply complementary
            theta = 0.98 * (theta + gyro_angle) + 0.02 * accel_angle
            
            # Render scene
            render_scene(offset_y, np.rad2deg(theta[0]), np.rad2deg(theta[1]))

            # Draw axes
            gl.glPushMatrix()
            gl.glTranslatef(0, 0, 1)
            draw_axes()
            gl.glPopMatrix()

        except KeyboardInterrupt:
            break

