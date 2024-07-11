import jetson.inference
import jetson.utils
import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Initialize jetson-inference object detection model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
# net = jetson.inference.detectNet("ssd-inception-v2", threshold=0.5)

def run_object_detection(color_frame, debug=False, all_classes=False):
    try:
        # Convert RealSense frame to CUDA image
        color_image = np.asanyarray(color_frame)
        # Ensure that the numpy array is not empty
        if color_image.size == 0:
            return []
        cuda_mem = jetson.utils.cudaFromNumpy(color_image)

        # Perform object detection
        detections = net.Detect(cuda_mem)
        print(f"detections = {detections}")
    except Exception as e:
        print(f"An error occurred during object detection: {e}")
        if not debug:
            return []
    if debug:
        # Draw bounding boxes and labels on the image
        for detection in detections:
            ID = detection.ClassID
            label = net.GetClassDesc(ID)
            if not all_classes and label not in ['orange', 'sports ball']:
                continue
            top = int(detection.Top)
            left = int(detection.Left)
            bottom = int(detection.Bottom)
            right = int(detection.Right)
            cv2.rectangle(color_image, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(color_image, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)
    else:
        # Show the live camera feed
        # Get the center of the bounding box, and average of the width and height of the bounding box and return it
        # This is the center of the object
        result = []
        for detection in detections:
            ID = detection.ClassID
            label = net.GetClassDesc(ID)
            if not all_classes and label not in ['orange', 'sports ball']:
                continue
            top = int(detection.Top)
            left = int(detection.Left)
            bottom = int(detection.Bottom)
            right = int(detection.Right)
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            width = right - left
            height = bottom - top
            result.append((center_x, center_y, width, height))
        # Convert the detections into dictionary
        for i in range(len(result)):
            result[i] = {'center_x': result[i][0], 'center_y': result[i][1], 'width': result[i][2], 'height': result[i][3]}
        return result

import camera
if __name__ == "__main__":
    camRS = camera.RealSenseCamera(640, 360)
    # camRS = camera.DepthAICamera(640, 360)
    while True:
        color_frame = camRS.read()[0]
        # cv2.imwrite("object_detection.png", color_frame)
        if color_frame is None:
            print("No frame received")
            continue
        # Toggle the all_classes flag here as needed
        run_object_detection(color_frame=color_frame, debug=True, all_classes=True)
# Cleanup
cv2.destroyAllWindows()
