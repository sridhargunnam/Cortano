# This doesn't work
# prompt 
# Modify the code in pub_sub_with_events.pt such that there is queue size of 2. When the data is published into the queue it is always writing a latest data into the queue. But there could be different process reading the queue. Therefore we need to acquire the resource first, and then use it. In this example camera_process will always be able to write the latest frame to one of the buffer. The subscriber in this case(object_detection) would have acquired a second item in the  queue that can be acquired. But once it is done processing it it will release that, and camera will be able to write to the other slot on the queue, and object_detection will be able to run on the latest frame that was read from camera. Is such an implementation possible? Modify the code to implement my request. Please think about various scenarios like deadlocks, and queues doing a shallow copy, etc. 
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
        self.net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

    def detect_objects(self, image):
        import jetson.utils
        import numpy as np
        try:
            color_image = np.asanyarray(image)
            if color_image.size == 0:
                print("The size of input image to ObjectDetectionML is 0")
                return []
            cuda_mem = jetson.utils.cudaFromNumpy(color_image)
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
                center_x = (left + right) / 2
                center_y = (top + bottom) / 2
                radius = ((right - left) + (bottom - top)) / 4
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
        response_data = client_socket.recv(1024)
        response = json.loads(response_data.decode())
        client_socket.close()
    except Exception as e:
        print(f"Failed to send command or receive response: {e}")
        pass
    return response

class PubSubWrapper:
    def __init__(self, queue_size=2):
        self.queues = {}
        self.locks = {}
        self.semaphores = {}
        self.queue_size = queue_size

    def create_queue(self, topic):
        if topic not in self.queues:
            self.queues[topic] = mp.Queue(self.queue_size)
            self.locks[topic] = mp.Lock()
            self.semaphores[topic] = mp.Semaphore(self.queue_size)

    def publish(self, topic, data):
        self.create_queue(topic)
        queue = self.queues[topic]
        lock = self.locks[topic]
        semaphore = self.semaphores[topic]
        
        with lock:
            if queue.full():
                queue.get()
            queue.put(data)
        semaphore.release()
        pub.sendMessage(topic, data=data)

    def subscribe(self, topic, callback):
        self.create_queue(topic)
        pub.subscribe(callback, topic)

    def get_queue(self, topic):
        self.create_queue(topic)
        return self.queues[topic], self.locks[topic], self.semaphores[topic]

def camera_process(pubsub, camera_ready_event):
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
            camera_ready_event.set()
        except Exception as e:
            print(f"Error in camera_process: {e}")
            cam.__del__()
            cam = camera.RealSenseCamera(640, 360)

def object_detection(pubsub, camera_topic, method, camera_ready_event):
    detector = method()
    camera_queue, camera_lock, camera_semaphore = pubsub.get_queue(camera_topic)
    while True:
        try:
            camera_ready_event.wait()
            camera_semaphore.acquire()
            with camera_lock:
                if not camera_queue.empty():
                    color_frame = camera_queue.get(timeout=1)
            detections = detector.detect_objects(color_frame)
            pubsub.publish('object/detections', detections)
            print(f"Object detection: Published {len(detections)} detections")
        except mp.queues.Empty:
            print("Object detection: Timeout waiting for camera frame")
        except Exception as e:
            print(f"Error in object_detection: {e}")

def object_mapper(pubsub, object_queue_topic, depth_queue_topic, intrinsics, camera_ready_event, detection_ready_event, mapping_ready_event):
    fx, fy, cx, cy, width, height, depth_scale = intrinsics.values()
    object_queue, object_lock, object_semaphore = pubsub.get_queue(object_queue_topic)
    depth_queue, depth_lock, depth_semaphore = pubsub.get_queue(depth_queue_topic)
    
    while True:
        try:
            detection_ready_event.wait()
            object_semaphore.acquire()
            depth_semaphore.acquire()
            with object_lock:
                if not object_queue.empty():
                    detections = object_queue.get(timeout=1)
            with depth_lock:
                if not depth_queue.empty():
                    depth_frame = depth_queue.get(timeout=1)
            camera_ready_event.clear()
            if detections is not None:
                for detection in detections:
                    center_x = detection['center_x']
                    center_y = detection['center_y']
                    depth_ = depth_frame[int(center_y)][int(center_x)] * depth_scale
                    x = 100 * (center_x - cx) * depth_ / fx
                    y = 100 * (center_y - cy) * depth_ / fy
                    z = 100 * depth_
                    pubsub.publish('location', {'x': x, 'y': y, 'z': z})
                    mapping_ready_event.set()
                    print(f'object is at {x}, {y}, {z}')
        except mp.queues.Empty:
            print("Object mapper: Timeout waiting for detections or depth frame")
        except Exception as e:
            print(f"Error in object_mapper: {e}")

import vex_serial as vex
def robotics_control(pubsub, location_queue_topic, mapping_ready_event):
    robot = vex.VexCortex("/dev/ttyUSB0")
    control = vex.VexControl(robot)
    control.stop_drive()
    location_queue, location_lock, location_semaphore = pubsub.get_queue(location_queue_topic)
    
    while True:
        try:
            mapping_ready_event.wait()
            location_semaphore.acquire()
            with location_lock:
                if not location_queue.empty():
                    location = location_queue.get(timeout=1)
            if location is not None:
                print(f"Robotics control: Processing location {location}")
                control.send_to_XY(location['x'], location['z'])
                mapping_ready_event.clear()
        except mp.queues.Empty:
            print("Robotics control: Timeout waiting for location data")
        except Exception as e:
            print(f"Robotics control: Error {e}")

def start_pipeline(queue_size=2):
    objectDetectionMethod = ObjectDetectionML
    pubsub = PubSubWrapper(queue_size=queue_size)

    manager = mp.Manager()
    intrinsicsRs = manager.dict(camera.read_calibration_data()['intrinsicsRs'])

    camera_ready_event = mp.Event()
    detection_ready_event = mp.Event()
    mapping_ready_event = mp.Event()

    camera_proc = mp.Process(target=camera_process, args=(pubsub, camera_ready_event))
    object_detection_proc = mp.Process(target=object_detection, args=(pubsub, 'camera/color', objectDetectionMethod, camera_ready_event))
    object_mapper_proc = mp.Process(target=object_mapper, args=(pubsub, 'object/detections', 'camera/depth', intrinsicsRs, camera_ready_event, detection_ready_event, mapping_ready_event))
    robotics_control_proc = mp.Process(target=robotics_control, args=(pubsub, 'location', mapping_ready_event))

    camera_proc.start()
    object_detection_proc.start()
    object_mapper_proc.start()
    robotics_control_proc.start()

    camera_proc.join()
    object_detection_proc.join()
    object_mapper_proc.join()
    robotics_control_proc.join()

if __name__ == "__main__":
    start_pipeline(queue_size=2)
