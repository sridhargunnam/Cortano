import jetson.inference
import jetson.utils
import pyrealsense2 as rs
import numpy as np
import cv2
import time


# Initialize jetson-inference object detection model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
# net = jetson.inference.detectNet("ssd-inception-v2", threshold=0.5)

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

def run_object_detection(color_frame, intrinsicsRS, rsCamToRobot, depth_frame, debug=False, rotate_image=False):
    try:
        # Convert RealSense frame to CUDA image
        color_image = np.asanyarray(color_frame)
                # Rotate the image if needed
        
        if rotate_image:
            color_image = cv2.rotate(color_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #ensure that the numpy array is not empty
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
        result = []
        for detection in detections:
            ID = detection.ClassID
            label = net.GetClassDesc(ID)
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
            cv2.putText(color_image, label, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # Annotate (x, z) coordinates
            # Calculate the center and radius
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            radius = ((right - left) + (bottom - top)) / 4  # Approximate radius
            object_wrt_frame = {'center_x': center_x, 'center_y': center_y, 'radius': radius}
            coordinates = get_object_wrt_robot(depth_frame, intrinsicsRS, object_wrt_frame)
                        # print(f'Ball w.r.t to camera x = {x}, y = {y}, z = {z}')
            robot2rsCam = np.linalg.inv(rsCamToRobot)
            object_pos_wrt_robot = robot2rsCam @ np.array([coordinates['x'], coordinates['y'], coordinates['z'], 1])
            annotation_text = f"x: {object_pos_wrt_robot[0]:.2f}, y: {object_pos_wrt_robot[1]:.2f}, z: {object_pos_wrt_robot[2]:.2f}"
            cv2.putText(color_image, annotation_text, (left, top-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)
    else:

        # Show the live camera feed
        #get the center of the bounding box, and average of the width and height of the bounding box and return it
        #this is the center of the object
        result = []
        for detection in detections:
            ID = detection.ClassID
            label = net.GetClassDesc(ID)
            if label not in ['orange', 'sports ball']:
                continue
            detection = detections[0]
            top = int(detection.Top)
            left = int(detection.Left)
            bottom = int(detection.Bottom)
            right = int(detection.Right)
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            width = right - left
            height = bottom - top
            result.append((center_x, center_y, width, height))
        # return detections
        # convert the detections into dictionary
        for i in range(len(result)):
            result[i] = {'center_x': result[i][0], 'center_y': result[i][1], 'width': result[i][2], 'height': result[i][3]}
        return result

import camera
if __name__ == "__main__":
    camRS = camera.RealSenseCamera(640,360) 
    # camRS = camera.DepthAICamera(640,360) 
    calib_data = camera.read_calibration_data()
    intrinsicsRS = calib_data['intrinsicsRs']
    rsCamToRobot = calib_data['rsCamToRobot']
    while True:
        color_frame, depth_frame = camRS.read()
        # cv2.imwrite("object_detection.png", color_frame)
        if color_frame is None or depth_frame is None:
            print("No frame received")
            continue
        run_object_detection(color_frame=color_frame,debug=True, rotate_image=False, intrinsicsRS=intrinsicsRS, depth_frame=depth_frame, rsCamToRobot= rsCamToRobot)
# Cleanup
cv2.destroyAllWindows()
