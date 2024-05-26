import multiprocessing as mp
import time
from pubsub import pub
import camera

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

def image_producer(queue):
    cam = camera.RealSenseCamera(640, 360)
    while True:
        try:
            color_frame, depth_frame = cam.read()
            if color_frame is None:
                continue
            capture_time = time.time()
            print("sending an image")
            queue.put((color_frame, capture_time))
        except Exception as e:
            print(f"Error in image_producer: {e}")
        # time.sleep(1)  # Adjust the sleep time as necessary

def object_detection_consumer(queue, method):
    detector = method()
    print("obj detection initialized")
    while True:
        try:
            if not queue.empty():
                image, capture_time = queue.get()
                print("received an image")
                detection_start_time = time.time()
                results = detector.detect_objects(image)
                detection_end_time = time.time()
                latency = detection_end_time - capture_time
                print(f"Detection results: {results}")
                pub.sendMessage("result_topic", results=results, latency=latency)
        except Exception as e:
            print(f"Error in object_detection_consumer: {e}")
        # time.sleep(0.1)  # Short sleep to prevent CPU hogging

def robot_control():
    def on_result_received(results, latency):
        if results:
            print(f"Ball coordinates: {results}, Latency: {latency:.4f} seconds")
        else:
            print("No ball detected.")
    pub.subscribe(on_result_received, "result_topic")

def main():
    # method = ObjectDetectionOpenCV  # Or ObjectDetectionML
    method = ObjectDetectionML
    queue = mp.Queue()
    
    producer_process = mp.Process(target=image_producer, args=(queue,))
    consumer_process = mp.Process(target=object_detection_consumer, args=(queue, method))
    robot_control_process = mp.Process(target=robot_control)

    consumer_process.start()
    producer_process.start()
    robot_control_process.start()

    producer_process.join()
    consumer_process.join()
    robot_control_process.join()

if __name__ == "__main__":
    main()
