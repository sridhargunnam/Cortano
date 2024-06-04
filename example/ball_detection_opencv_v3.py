import cv2
import numpy as np
import time
import camera
from queue import Empty, Full, Queue

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

def distance(p1, p2):
    """Calculate distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def run_object_detection(intrinsicsRS, rsCamToRobot, image, depth, capture_time, depth_scale=1.0, debug=True, min_distance=70):
    result = ball_detection(image, depth, intrinsicsRS, rsCamToRobot, debug, min_distance)
    detection_time = time.time()
    latency = detection_time - capture_time
    return {'result': result, 'latency': latency}

def ball_detection(image, depth, intrinsicsRS, rsCamToRobot, debug=True, min_distance=70):  # Added min_distance parameter
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
        
        for r in result:
            # Annotate (x, z) coordinates
            # Calculate the center and radius
            center_x = int(r['center_x'])  # Cast to int
            center_y = int(r['center_y'])  # Cast to int
            radius = (r['width'] + r['height']) / 2  # Approximate radius
            object_wrt_frame = {'center_x': center_x, 'center_y': center_y, 'radius': radius}
            coordinates = get_object_wrt_robot(depth, intrinsicsRS, object_wrt_frame)
            robot2rsCam = np.linalg.inv(rsCamToRobot)
            object_pos_wrt_robot = robot2rsCam @ np.array([coordinates['x'], coordinates['y'], coordinates['z'], 1])
            annotation_text = f"x: {object_pos_wrt_robot[0]:.2f}, y: {object_pos_wrt_robot[1]:.2f}, z: {object_pos_wrt_robot[2]:.2f}"
            cv2.putText(image, annotation_text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Use casted values
        
        if downscale != 1:
            image = cv2.resize(image, (0, 0), fx=downscale, fy=downscale)
        
        cv2.imshow('Color', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit(0)
    
    return result

def main():
    cam = camera.RealSenseCamera(640, 360)

    calib_data = camera.read_calibration_data()
    intrinsicsRS = calib_data['intrinsicsRs']
    rsCamToRobot = calib_data['rsCamToRobot']

    try:
        while True:
            image, depth = cam.read()
            capture_time = time.time()
            
            if image is None:
                continue
            
            detection_result = run_object_detection(intrinsicsRS, rsCamToRobot, image, depth, capture_time)
            # print(f"Result: {detection_result['result']}, Latency: {detection_result['latency']:.4f} seconds")

    except KeyboardInterrupt:
        print("Program terminated by user.")

if __name__ == "__main__":
    downscale = 1  # or any other value for testing
    main()
