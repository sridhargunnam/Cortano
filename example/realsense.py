import jetson.inference
import jetson.utils
import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Initialize RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# Initialize jetson-inference object detection model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# For FPS calculation
frame_count = 0
start_time = time.time()
fps = 1

while True:
    # Get frames from the RealSense camera
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    # Convert RealSense frame to CUDA image
    color_image = np.asanyarray(color_frame.get_data())
    cuda_mem = jetson.utils.cudaFromNumpy(color_image)

    # Perform object detection
    detections = net.Detect(cuda_mem)
    
    # Draw bounding boxes and labels on the image
    for detection in detections:
        ID = detection.ClassID
        label = net.GetClassDesc(ID)
        if label not in ['orange', 'sports ball']:
            continue
        top = int(detection.Top)
        left = int(detection.Left)
        bottom = int(detection.Bottom)
        right = int(detection.Right)
        cv2.rectangle(color_image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(color_image, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    # Calculate and display FPS
    frame_count += 1
    if (time.time() - start_time) > 1:
        fps = frame_count / (time.time() - start_time)
        frame_count = 0
        start_time = time.time()

    cv2.putText(color_image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the live camera feed
    cv2.imshow('RealSense', color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()
