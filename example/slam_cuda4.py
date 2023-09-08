#########################################################
print("the script doesn't work yet")
exit(0)
import numpy as np
import cupoch as cph
from cortano import RemoteInterface
import numpy as np
from datetime import datetime
import multiprocessing
import queue
# import camera
import pyrealsense2 as rs
import cv2

#parse arguments
import argparse
parser = argparse.ArgumentParser(description='Cortano SLAM')
parser.add_argument('--runLocal', type=bool, default=True, help='run on local machine or on remote robot')
args = parser.parse_args()


cph.initialize_allocator(cph.PoolAllocation, 1000000000)

# depth
fx = 460.92495728   # FOV(x) -> depth2xyz -> focal length (x)
fy = 460.85058594   # FOV(y) -> depth2xyz -> focal length (y)
cx = 315.10949707   # 640 (width) 320
cy = 176.72598267   # 360 (height) 180

# # check if open3d is using cuda
# import open3d as o3d
# if o3d.core.cuda.device_count() > 0:
#     print("Open3D is using CUDA")
# else:
#     print("Open3D is not using CUDA")

def producer(queue, isRunning, cam, num_productions):
    print("producer isRunning: " + str(isRunning.Value) + " queue size: " + str(queue.qsize()))
    if not args.runLocal:
        robot = RemoteInterface("192.168.68.68")        

    
    # while isRunning:
    for _ in range(num_productions):
        print("inside producer, current queue size: ", queue.qsize())
        if not args.runLocal:
            robot.update()
            valid, c, d, _ = robot.read() # FIXME add the valid flag in the read function, and remove the override
            valid = True
        else:
            print("inside producer capturing data")
            valid, c, d = cam.capture()
            if valid:
                print("captured valid data")
            else:
                print("invalid data, will try to capture again")
            continue
        try:
            item = (c,d)
            queue.put_nowait(item)
        except Exception as e:
            # print("queue is full, dropping data")
            pass
    
    print("producer ends")
    isRunning.Value = 0

    
def consumer(queue, isRunning, num_productions):
    print("consumer isRunning: " + str(isRunning.Value) + " queue size: " + str(queue.qsize()))
    prev_rgbd_image = None

    camera_intrinsics = cph.camera.PinholeCameraIntrinsic(640,360,fx,fy,cx,cy)
    option = cph.odometry.OdometryOption()
    option.min_depth = 0.30
    option.max_depth = 4
    # print(dir(cph.odometry.OdometryOption))
    cur_trans = np.identity(4)

    # while isRunning:
    for _ in range(num_productions):
        print("inside consumer, current queue size: ", queue.qsize())
        try:
            dt = datetime.now()

            color, depth = queue.get()

            # color = color.astype(np.float32)
            depth = depth.astype(np.float32)

            color = cph.geometry.Image(color)
            depth = cph.geometry.Image(depth)

            rgbd = cph.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale = 1000
            )

            if not prev_rgbd_image is None:
                res, odo_trans, _ = cph.odometry.compute_rgbd_odometry(
                    rgbd,
                    prev_rgbd_image,
                    camera_intrinsics,
                    np.identity(4),
                    cph.odometry.RGBDOdometryJacobianFromHybridTerm(),
                    option,
                )

                if res:
                    cur_trans = cur_trans @ odo_trans
                    print(cur_trans[:3,3])

            prev_rgbd_image = rgbd
            process_time = datetime.now() - dt
            print("FPS: " + str(1 / process_time.total_seconds()))
        except:
            # print("queue is empty")
            pass
    print("consumer ends")
    isRunning.Value = 0


def getRecorededFrames(queue, isRunning, num_productions):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file('/home/sgunnam/Documents/20230906_080949.bag')
    pipeline.start(config)
    for _ in range(num_productions):
        try:
            # Wait for a new frame
            frames = pipeline.wait_for_frames()
            if not frames:
                break  # No more frames, exit the loop

            # get depth and color frames
            for frame in frames:
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                dWidth = depth_frame.get_width()
                dHeight = depth_frame.get_height()
                cHeight = color_frame.get_height()
                cWidth = color_frame.get_width()
                #view the depth image and color image using openCV in a same window
                #create a numpy array with the correct dimensions from the raw data
                depth_array = np.asanyarray(depth_frame.get_data())
                color_array = np.asanyarray(color_frame.get_data())
        except:
            break
    pipeline.stop()


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn', force=True)
    num_productions = 500
    queue = multiprocessing.Queue(maxsize=1)
    isRunning = multiprocessing.Value('i', 1)
    producer_p = multiprocessing.Process(target=producer, args=(queue,isRunning, cameraSetup(), num_productions))
    # consumer_p = multiprocessing.Process(target=consumer, args=(queue,isRunning, num_productions))

    producer_p.start()
    # consumer_p.start()
    # producer(queue, isRunning)
    # consumer(queue, isRunning)



    producer_p.join()
    # consumer_p.join()
    print("Good bye!")