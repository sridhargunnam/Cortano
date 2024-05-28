import multiprocessing as mp
from pubsub import pub
import time
import camera
import cv2
import numpy as np

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

import socket
import json

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
    def __init__(self):
        self.queues = {}

    def create_queue(self, topic):
        if topic not in self.queues:
            self.queues[topic] = mp.Queue()

    def publish(self, topic, data):
        self.create_queue(topic)
        self.queues[topic].put(data)
        pub.sendMessage(topic, data=data)

    def subscribe(self, topic, callback):
        self.create_queue(topic)
        pub.subscribe(callback, topic)

    def get_queue(self, topic):
        self.create_queue(topic)
        return self.queues[topic]

def camera_process(pubsub):
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
        except Exception as e:
            print(f"Error in camera_process: {e}")
            # Consider resetting the camera or reinitializing
            cam.__del__()  # Clean up resources
            cam = camera.RealSenseCamera(640, 360)  # Reinitialize

def april_tag_detector(pubsub, camera_queue):
    while True:
        try:
            color_frame = camera_queue.get(timeout=5)
            detections = f"AprilTag Detected on frame {color_frame}"
            pubsub.publish('april_tag/detections', detections)
        except mp.queues.Empty:
            print("AprilTag detector: Timeout waiting for camera frame")
        except Exception as e:
            print(f"Error in april_tag_detector: {e}")

def object_detection(pubsub, camera_queue, method):
    detector = method()
    while True:
        try:
            color_frame = camera_queue.get(timeout=5)
            detections = detector.detect_objects(color_frame)
            pubsub.publish('object/detections', detections)
            print(f"Object detection: Published {len(detections)} detections")
        except mp.queues.Empty:
            print("Object detection: Timeout waiting for camera frame")
        except Exception as e:
            print(f"Error in object_detection: {e}")

def object_mapper(pubsub, object_queue, depth_queue, intrinsics):
    fx, fy, cx, cy, width, height, depth_scale = intrinsics.values()
    while True:
        try:
            detections = object_queue.get(timeout=5)
            depth_frame = depth_queue.get(timeout=5)
            print(f"Object mapper: Processing {len(detections)} detections")
            for detection in detections:
                center_x = detection['center_x']
                center_y = detection['center_y']
                depth_ = depth_frame[int(center_y)][int(center_x)]
                depth_ = depth_ * depth_scale
                # get the x, y, z from image frame to camera frame
                x = center_x
                y = center_y
                if center_x == 0 and center_y == 0:
                    print("abnormality ")
                    exit(0)
                x = 100 * (x - cx) * depth_ / fx  # multiply by 100 to convert to centimeters
                y = 100 * (y - cy) * depth_ / fy
                z = 100 * depth_
                # if abs(x) != 0.0 and abs(y) != 0.0 and abs(z) != 0.0:  # sometime the NN detections can be zeros
                pubsub.publish('location', {'x': x, 'y': y, 'z': z})
                print(f'object is at {x}, {y}, {z}')
        except mp.queues.Empty:
            print("Object mapper: Timeout waiting for detections or depth frame")
        except Exception as e:
            print(f"Error in object_mapper: {e}")

import vex_serial as vex
def robotics_control(pubsub, location_queue):
    robot = vex.VexCortex("/dev/ttyUSB0")
    control = vex.VexControl(robot)
    control.stop_drive()
    while True:
        try:
            location = location_queue.get(timeout=5)
            print(f"Robotics control: Processing location {location}")
            control.send_to_XY(location['x'], location['z'])
            # control_commands = f"Control commands for {location}"
            # print(control_commands)
        except mp.queues.Empty:
            control.stop_drive()
            print("Robotics control: Timeout waiting for location data")
        except Exception as e:
            control.stop_drive()
            print(f"Error in robotics_control: {e}")
        

def start_pipeline():
    # objectDetectionMethod = ObjectDetectionOpenCV
    objectDetectionMethod = ObjectDetectionML
    pubsub = PubSubWrapper()

    camera_queue_color = pubsub.get_queue('camera/color')
    camera_queue_depth = pubsub.get_queue('camera/depth')
    object_detection_queue = pubsub.get_queue('object/detections')
    # april_tag_queue = pubsub.get_queue('april_tag/detections')
    location_queue = pubsub.get_queue('location')
    manager = mp.Manager()
    intrinsicsRs = manager.dict(camera.read_calibration_data()['intrinsicsRs'])

    camera_proc = mp.Process(target=camera_process, args=(pubsub,))
    # april_tag_proc = mp.Process(target=april_tag_detector, args=(pubsub, camera_queue_color))
    object_detection_proc = mp.Process(target=object_detection, args=(pubsub, camera_queue_color, objectDetectionMethod))
    object_mapper_proc = mp.Process(target=object_mapper, args=(pubsub, object_detection_queue, camera_queue_depth, intrinsicsRs))
    robotics_control_proc = mp.Process(target=robotics_control, args=(pubsub, location_queue))

    camera_proc.start()
    # april_tag_proc.start()
    object_detection_proc.start()
    object_mapper_proc.start()
    robotics_control_proc.start()

    camera_proc.join()
    # april_tag_proc.join()
    object_detection_proc.join()
    object_mapper_proc.join()
    robotics_control_proc.join()

if __name__ == "__main__":
    start_pipeline()
