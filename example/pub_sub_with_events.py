import multiprocessing as mp
from pubsub import pub
import time
import camera
import cv2
import numpy as np
import socket
import json

class ObjectDetectionBase:
    def __init__(self):
        pass

    def detect_objects(self, image):
        raise NotImplementedError("Must be implemented by subclass")

class ObjectDetectionOpenCV(ObjectDetectionBase):
    def detect_objects(self, image):
        # OpenCV detection logic
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > 30]
        result = []
        for contour in contours:
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            result.append({'center_x': center_x, 'center_y': center_y, 'radius': radius})
        return result

class ObjectDetectionML(ObjectDetectionBase):
    def __init__(self):
        import jetson.inference
        import jetson.utils
        # Initialize the Jetson inference model
        self.net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

    def detect_objects(self, image):
        import jetson.utils
        import numpy as np
        try:
            # Convert RealSense frame to CUDA image
            color_image = np.asanyarray(image)
            # Ensure that the numpy array is not empty
            if color_image.size == 0:
                print("The size of input image to ObjectDetectionML is 0")
                return []
            cuda_mem = jetson.utils.cudaFromNumpy(color_image)
            # Perform object detection
            detections = self.net.Detect(cuda_mem)
            result = []
            for detection in detections:
                ID = detection.ClassID
                label = self.net.GetClassDesc(ID)
                if label not in ['orange', 'sports ball']:
                    continue
                top = int(detection.Top)
                left = int(detection.Left)
                bottom = int(detection.Bottom)
                right = int(detection.Right)
                # Calculate the center and radius
                center_x = (left + right) / 2
                center_y = (top + bottom) / 2
                radius = ((right - left) + (bottom - top)) / 4  # Approximate radius
                result.append({'center_x': center_x, 'center_y': center_y, 'radius': radius})
            return result
        except Exception as e:
            print(f"An error occurred during object detection: {e}")
            return []

def send_command(command, args=None):
    response = None 
    try:
        print("sending ", command, args)
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 6000))
        command_data = json.dumps({'command': command, 'args': args})
        client_socket.send(command_data.encode())
        # Waiting for response
        response_data = client_socket.recv(1024)
        response = json.loads(response_data.decode())
        client_socket.close()
    except Exception as e:
        print(f"Failed to send command or receive response: {e}")
        pass
    return response

class PubSubWrapper:
    def __init__(self, queue_size=1):
        self.queues = {}
        self.queue_size = queue_size

    def create_queue(self, topic, queue_size=1):
        if topic not in self.queues:
            self.queues[topic] = mp.Queue(queue_size)

    def publish(self, topic, data):
        self.create_queue(topic)
        queue = self.queues[topic]
        if queue.full():
            queue.get()  # Remove the oldest item
        queue.put(data)
        pub.sendMessage(topic, data=data)

    def subscribe(self, topic, callback):
        self.create_queue(topic)
        pub.subscribe(callback, topic)

    def get_queue(self, topic, queue_size=1):
        self.create_queue(topic, queue_size)
        return self.queues[topic]

def camera_process(pubsub, camera_ready_event, detection_ready_event):
    cam = camera.RealSenseCamera(640, 360)
    while True:
        try:
            color_frame, depth_frame = cam.read()
            if color_frame is None:
                print('color_frame is none')
                continue
            print("Camera process: sending an image")
            pubsub.publish('camera/color', color_frame)
            pubsub.publish('camera/depth', depth_frame)
            camera_ready_event.set()  # Signal that the frame is ready
            # detection_ready_event.wait()  # Wait for detection to complete
            # detection_ready_event.clear()  # Reset detection event
        except Exception as e:
            print(f"Error in camera_process: {e}")
            cam.__del__()  # Clean up resources
            cam = camera.RealSenseCamera(640, 360)  # Reinitialize
# import tag as tag
    # atag = tag.ATag(camera_params=)
def object_detection(pubsub, camera_queue, method, camera_ready_event, detection_ready_event):
    detector = method()
    while True:
        try:
            camera_ready_event.wait()  # Wait for the camera to produce a frame
            while not camera_queue.empty():
                color_frame = camera_queue.get(timeout=1)  # Get the most recent frame
            detections = detector.detect_objects(color_frame)
            # aprilTags  = tag
            pubsub.publish('object/detections', detections)
            detection_ready_event.set()  # Signal that detection is done
            print(f"Object detection: Published {len(detections)} detections")
        except mp.queues.Empty:
            print("Object detection: Timeout waiting for camera frame")
        except Exception as e:
            print(f"Error in object_detection: {e}")

def get_object_wrt_robot(depth_frame, intrinsics, detection):
    fx, fy, cx, cy, width, height, depth_scale = intrinsics.values()
    try:
        if detection is not None:
            center_x = detection['center_x']
            center_y = detection['center_y']
            depth_ = depth_frame[int(center_y)][int(center_x)]
            depth_ = depth_ * depth_scale
            x = center_x
            y = center_y
            if center_x == 0 and center_y == 0:
                print("Abnormality in NN detections x=0, y=0")
            else:
                x = 100 * (x - cx) * depth_ / fx  # Multiply by 100 to convert to centimeters
                y = 100 * (y - cy) * depth_ / fy
                z = 100 * depth_
                object_coordinates = {'x': x, 'y': y, 'z': z}
                print(f'Object is at (x, y, z): ({x}, {y}, {z})')
                return object_coordinates
    except Exception as e:
        print(f"Error in object_mapper: {e}")

def object_mapper(pubsub, object_queue, depth_queue, camera_ready_event, detection_ready_event, mapping_ready_event):
    # fx, fy, cx, cy, width, height, depth_scale = calib_data['intrinsicsRs'].values()
    calib_data = camera.read_calibration_data()
    # print(calib_data.keys())
    intrinsicsRs = calib_data['intrinsicsRs']
    rsCamToRobot = calib_data['rsCamToRobot']
    # exit(0)
    while True:
        try:
            detection_ready_event.wait()  # Wait for detection to complete
            while not object_queue.empty():
                detections = object_queue.get(timeout=1)  # Get the most recent detections
                detection_ready_event.clear()  # clear 
                print(f"Object mapper: Processing {len(detections)} detections")
            while not depth_queue.empty():
                depth_frame = depth_queue.get(timeout=1)  #TODO Get the most recent depth frame
            camera_ready_event.clear()  
            if detections is not None:
                for detection in detections:
                    object_wrt_frame = detection #{'center_x': center_x, 'center_y': center_y, 'radius': radius}
                    coordinates = get_object_wrt_robot(depth_frame, intrinsicsRs, object_wrt_frame)
                    robot2rsCam = np.linalg.inv(rsCamToRobot)
                    object_pos_wrt_robot = robot2rsCam @ np.array([coordinates['x'], coordinates['y'], coordinates['z'], 1])
                    if object_pos_wrt_robot[0] == 0 and object_pos_wrt_robot[2] == 0:
                        print("abnormality in NN detections x=0, y=0")
                        # exit(0)
                    else:
                        print(f"x: {object_pos_wrt_robot[0]:.2f}, y: {object_pos_wrt_robot[1]:.2f}, z: {object_pos_wrt_robot[2]:.2f}")
                        pubsub.publish('location', {'x': object_pos_wrt_robot[0], 'y': object_pos_wrt_robot[2], 'z': object_pos_wrt_robot[1]})
                        mapping_ready_event.set()  # Signal that mapping is done
                        print(f'object is at {object_pos_wrt_robot[0]}, {object_pos_wrt_robot[1]}, {object_pos_wrt_robot[2]}')
        except mp.queues.Empty:
            print("Object mapper: Timeout waiting for detections or depth frame")
        except Exception as e:
            print(f"Error in object_mapper: {e}")

import vex_serial as vex
def robotics_control(pubsub, location_queue, mapping_ready_event):
    # time.sleep(5)
    # try:
    robot = vex.VexCortex("/dev/ttyUSB0")
    control = vex.VexControl(robot)
    control.stop_drive()
    # except Exception as e:
    #     print(f"Robotics control: Error {e}")
        # exit(0)
    while True:
        try:
            mapping_ready_event.wait()  # Wait for mapping to complete
            location = None
            while not location_queue.empty():
                # queue_size = location_queue.qsize()  # Get the approximate size of the queue
                # print(f"Robotics control: location_queue size is {queue_size}")
                location = location_queue.get(timeout=1)  # Get the most recent location
            if location is not None:
                mapping_ready_event.clear()  
                print(f"Robotics control: Processing location {location}")
                control.send_to_XY(location['x'], location['z'])
        except mp.queues.Empty:
            # control.stop_drive()
            print("Robotics control: Timeout waiting for location data")
        except Exception as e:
            # control.stop_drive()
            print(f"Robotics control: Error {e}")
        # time.sleep(0.1)


def start_pipeline(queue_size=1):
    objectDetectionMethod = ObjectDetectionML
    pubsub = PubSubWrapper(queue_size=queue_size)

    camera_queue_color = pubsub.get_queue('camera/color')
    camera_queue_depth = pubsub.get_queue('camera/depth')
    object_detection_queue = pubsub.get_queue('object/detections')
    location_queue = pubsub.get_queue('location', 1000)
    manager = mp.Manager()
    # calib_data = camera.read_calibration_data()
    # calibData = manager.dict(calib_data)
    # print(calib_data['intrinsicsRs'])
    # print(f"calib_data {type(calibData)}")
    # exit(0)

    camera_ready_event = mp.Event()
    detection_ready_event = mp.Event()
    mapping_ready_event = mp.Event()

    camera_proc = mp.Process(target=camera_process, args=(pubsub, camera_ready_event, detection_ready_event))
    object_detection_proc = mp.Process(target=object_detection, args=(pubsub, camera_queue_color, objectDetectionMethod, camera_ready_event, detection_ready_event))
    object_mapper_proc = mp.Process(target=object_mapper, args=(pubsub, object_detection_queue, camera_queue_depth, camera_ready_event, detection_ready_event, mapping_ready_event))
    robotics_control_proc = mp.Process(target=robotics_control, args=(pubsub, location_queue, mapping_ready_event))

    camera_proc.start()
    object_detection_proc.start()
    object_mapper_proc.start()
    robotics_control_proc.start()

    camera_proc.join()
    object_detection_proc.join()
    object_mapper_proc.join()
    robotics_control_proc.join()

if __name__ == "__main__":
    start_pipeline(queue_size=1)
