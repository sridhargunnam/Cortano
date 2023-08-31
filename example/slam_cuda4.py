import numpy as np
import cupoch as cph
from cortano import RemoteInterface
import numpy as np
from datetime import datetime
import multiprocessing
import queue


cph.initialize_allocator(cph.PoolAllocation, 1000000000)

# depth
fx = 450.062347412109
fy = 450.062347412109
cx = 315.441986083984
cy = 190.89762878418

def producer(queue, isRunning):
    robot = RemoteInterface("192.168.68.68")

    while isRunning:
        robot.update()
        c, d, _ = robot.read()
        try:
            item = (c,d)
            queue.put_nowait(item)
        except Exception as e:
            # print("queue is full, dropping data")
            pass
    
    print("producer ends")
    isRunning = 0

    
def consumer(queue, isRunning):
    prev_rgbd_image = None

    camera_intrinsics = cph.camera.PinholeCameraIntrinsic(640,360,fx,fy,cx,cy)
    option = cph.odometry.OdometryOption()
    option.min_depth = 0.30
    option.max_depth = 4
    # print(dir(cph.odometry.OdometryOption))
    cur_trans = np.identity(4)

    while isRunning:
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
    
    print("consumer ends")
    isRunning = 0



if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn', force=True)
    queue = multiprocessing.Queue(maxsize=1)
    isRunning = multiprocessing.Value('i', 1)
    producer_p = multiprocessing.Process(target=producer, args=(queue,isRunning))
    # consumer_p = multiprocessing.Process(target=consumer, args=(queue,isRunning))

    producer_p.start()
    # consumer_p.start()
    # producer(queue, isRunning)
    consumer(queue, isRunning)



    # consumer_p.join()
    producer_p.join()
    print("Good bye!")