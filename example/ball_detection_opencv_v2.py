import cv2
import numpy as np
import time
import camera
import multiprocessing as mp
from queue import Empty, Full

def distance(p1, p2):
    """Calculate distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def run_object_detection_mp(rs_queue_ocv, rs_ocv_ball_detection_queue, depth_scale=1.0):
    while True:
        item = None
        try:
            while not rs_queue_ocv.empty():  # Clear any old images to get the latest one
                item = rs_queue_ocv.get_nowait()
        except Empty:
            continue
        
        if item is None or item['image'] is None:
            continue
        
        image, capture_time = item['image'], item['capture_time']
        result = ball_detection(image)
        detection_time = time.time()
        latency = detection_time - capture_time
        
        try:
            while not rs_ocv_ball_detection_queue.empty():  # Clear any old results to keep only the latest
                rs_ocv_ball_detection_queue.get_nowait()
            rs_ocv_ball_detection_queue.put_nowait({'result': result, 'latency': latency})
        except Full:
            continue

def ball_detection(image, debug=True, min_distance=30):  # Added min_distance parameter
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 30]
    
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
    
    unique_contours = []
    for i, c1 in enumerate(centroids):
        if all(distance(c1, c2) >= min_distance for j, c2 in enumerate(centroids) if i != j):
            unique_contours.append(contours[i])

    result = []
    for i in range(len(unique_contours)):
        (center_x, center_y), radius = cv2.minEnclosingCircle(unique_contours[i])
        result.append({'center_x': center_x, 'center_y': center_y, 'width': radius, 'height': radius})

    if debug:
        for contour in unique_contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            cv2.circle(image, center, int(radius), (0, 255, 0), 2)
        
        if downscale != 1:
            image = cv2.resize(image, (0, 0), fx=downscale, fy=downscale)
        
        cv2.imshow('Color', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit(0)
    
    return result

def main():
    cam = camera.RealSenseCamera(640, 360)
    rs_queue_ocv = mp.Queue(maxsize=2)
    rs_ocv_ball_detection_queue = mp.Queue(maxsize=2)

    process = mp.Process(target=run_object_detection_mp, args=(rs_queue_ocv, rs_ocv_ball_detection_queue))
    process.start()

    try:
        while True:
            image = cam.read()[0]
            capture_time = time.time()
            
            if image is None:
                continue
            
            try:
                if rs_queue_ocv.full():  # Remove the oldest image if the queue is full
                    rs_queue_ocv.get_nowait()
                rs_queue_ocv.put_nowait({'image': image, 'capture_time': capture_time})
            except Full:
                continue
            except Empty:
                continue

            try:
                item = None
                while not rs_ocv_ball_detection_queue.empty():  # Get the latest result
                    item = rs_ocv_ball_detection_queue.get_nowait()
                if item is not None:
                    result, latency = item['result'], item['latency']
                    print(f"Result: {result}, Latency: {latency:.4f} seconds")
            except Empty:
                continue

    except KeyboardInterrupt:
        process.terminate()
        process.join()

if __name__ == "__main__":
    downscale = 1  # or any other value for testing
    main()
