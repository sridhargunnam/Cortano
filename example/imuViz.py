import pyrealsense2 as rs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import config
config = config.Config()
# from filterpy.monte_carlo import resample

def resample(particles, weights):
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0  # Ensure sum is exactly one
    indexes = np.searchsorted(cumulative_sum, np.random.uniform(size=len(weights)))
    resampled_particles = [particles[i] for i in indexes]
    return resampled_particles


class Particle:
    def __init__(self, position, orientation, velocity, weight):
        self.position = np.array(position, dtype=np.float64)
        self.orientation = np.array(orientation, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.weight = weight

def motion_update(particles, accel, gyro, dt):
    for particle in particles:
        # Update velocity based on acceleration
        particle.velocity += accel * dt

        # Update position based on velocity
        particle.position += particle.velocity * dt

        # Update orientation based on gyroscope data
        particle.orientation += gyro * dt

def estimate_state(particles):
    position = np.mean([p.position for p in particles], axis=0)
    orientation = np.mean([p.orientation for p in particles], axis=0)
    return position, orientation

# Initialize particles
num_particles = 1000
particles = [Particle(position=[0.0, 0.0, 0.0], orientation=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], weight=1.0) for _ in range(num_particles)]


# Read calibration file and get transformation matrix
def readCalibrationFile(path=config.CALIB_PATH):
    calib = np.loadtxt(path, delimiter=",")
    rsCamToRobot, daiCamToRobot = calib[:4,:], calib[4:,:]
    return np.linalg.inv(rsCamToRobot), daiCamToRobot

robotToCam, _ = readCalibrationFile()

# Initialize RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
pipeline.start(config)

# Prepare the matplotlib plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', autoscale_on=False)
plt.ion()  # Interactive mode

# Time interval
dt = 1.0 / 100  # Assuming 100 Hz IMU data rate

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)

        if not accel_frame or not gyro_frame:
            continue

        # Transform IMU data to robot frame
        accel = np.dot(robotToCam[:3, :3], np.array([accel_frame.as_motion_frame().get_motion_data().x,
                                                     accel_frame.as_motion_frame().get_motion_data().y,
                                                     accel_frame.as_motion_frame().get_motion_data().z], dtype=np.float64))

        gyro = np.dot(robotToCam[:3, :3], np.array([gyro_frame.as_motion_frame().get_motion_data().x,
                                                    gyro_frame.as_motion_frame().get_motion_data().y,
                                                    gyro_frame.as_motion_frame().get_motion_data().z], dtype=np.float64))

        # Update particles
        motion_update(particles, accel, gyro, dt)

        # Resample particles
        weights = np.array([particle.weight for particle in particles])
        particles = resample(particles, weights)

        # Estimate state
        position, orientation = estimate_state(particles)

        # Clear the plot and redraw
        ax.clear()
        ax.scatter(position[0], position[1], position[2], color='r')
        ax.quiver(position[0], position[1], position[2], orientation[0], orientation[1], orientation[2], length=0.1, color='b')

        plt.draw()
        plt.pause(0.001)

except Exception as e:
    print(e)

finally:
    pipeline.stop()
    plt.ioff()
