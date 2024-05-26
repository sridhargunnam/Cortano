# object_detection_base.py
class ObjectDetectionBase:
    def __init__(self):
        pass
    
    def detect_objects(self, image):
        raise NotImplementedError("Must be implemented by subclass")

class ObjectDetectionOpenCV(ObjectDetectionBase):
    def detect_objects(self, image):
        import cv2
        import numpy as np

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
    def detect_objects(self, image):
        import numpy as np
        # ML detection logic (placeholder)
        # Replace with actual model prediction
        result = [{'center_x': np.random.randint(0, image.shape[1]),
                   'center_y': np.random.randint(0, image.shape[0]),
                   'radius': np.random.randint(10, 50)}]
        return result


import cv2
import time
import queue
import multiprocessing as mp
# from object_detection_base import ObjectDetectionOpenCV, ObjectDetectionML
import camera  # Assuming you have a custom camera module

def image_producer(image_queue):
    cam = camera.RealSenseCamera(640, 360)
    while True:
        color_frame, depth_frame  = cam.read()
        capture_time = time.time()
        if color_frame is None:
            continue
        try:
            if not image_queue.full():  # Only put new image if there is space
                image_queue.put_nowait({'image': color_frame, 'capture_time': capture_time})
            else:
                image_queue.get_nowait()  # Remove the oldest image
                image_queue.put_nowait({'image': color_frame, 'capture_time': capture_time})
        except queue.Empty:
            pass
        except queue.Full:
            continue

def object_detection_consumer(image_queue, result_queue, method):
    detector = method()
    while True:
        frame_data = None
        try:
            if not image_queue.empty():
                frame_data = image_queue.get_nowait()
        except queue.Empty:
            continue
        
        if frame_data is None or 'image' not in frame_data:
            continue
        
        frame = frame_data['image']
        capture_time = frame_data['capture_time']
        
        # if not isinstance(frame, np.ndarray):
        #     print("Invalid frame type:", type(frame))
        #     continue
        
        results = detector.detect_objects(frame)
        detection_end_time = time.time()
        latency = detection_end_time - capture_time
        if not result_queue.full():
            result_queue.put((results, latency))

def robot_control(result_queue):
    while True:
        try:
            while not result_queue.empty():  # Get the latest result
                results, latency = result_queue.get_nowait()
                if results:
                    print(f"Ball coordinates: {results}, Latency: {latency:.4f} seconds")
                else:
                    print("No ball detected.")
        except queue.Empty:
            continue

def main():
    image_queue = mp.Queue(maxsize=2)
    result_queue = mp.Queue(maxsize=2)

    method = ObjectDetectionOpenCV  # Or ObjectDetectionML
    producer_process = mp.Process(target=image_producer, args=(image_queue,))
    consumer_process = mp.Process(target=object_detection_consumer, args=(image_queue, result_queue, method))
    robot_control_process = mp.Process(target=robot_control, args=(result_queue,))

    producer_process.start()
    consumer_process.start()
    robot_control_process.start()

    producer_process.join()
    consumer_process.join()
    robot_control_process.join()

from pubsub import pub


if __name__ == "__main__":
    main()

