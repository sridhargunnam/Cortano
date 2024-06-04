import cv2
import numpy as np
import socket
import json
import camera
import vex_serial as vex


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
                x = 100 * (x - cx) * depth_ / fx
                y = 100 * (y - cy) * depth_ / fy
                z = 100 * depth_
                object_coordinates = {'x': x, 'y': y, 'z': z}
                print(f'Object is at (x, y, z): ({x}, {y}, {z})')
                return object_coordinates
    except Exception as e:
        print(f"Error in get_object_wrt_robot: {e}")

def main():
    cam = camera.RealSenseCamera(640, 360)
    objectDetectionMethod = ObjectDetectionML()
    objectDetectionMethodAlternate = ObjectDetectionOpenCV()
    enable_opencv_when_ML_is_not_available = True 
    calib_data = camera.read_calibration_data()
    intrinsicsRs = calib_data['intrinsicsRs']
    rsCamToRobot = calib_data['rsCamToRobot']
    robot = vex.VexCortex("/dev/ttyUSB0")
    control = vex.VexControl(robot)
    control.stop_drive()
    random_dir = 'backward'
    drive_time = 0.05
    while True:
        drive_time = 0.05
        try:
            color_frame, depth_frame = cam.read()
            if color_frame is None or depth_frame is None:
                print('color_frame or depth frame is none')
                continue

            detections = objectDetectionMethod.detect_objects(color_frame)
            if len(detections) == 0 and enable_opencv_when_ML_is_not_available:
                detections = objectDetectionMethodAlternate.detect_objects(color_frame)
            # while len(detections) == 0:
            #     print("objects not found")
            #     control.drive([random_dir, 30, drive_time])
            #     if random_dir == 'backward':
            #         random_dir = 'forward'
            #     else:
            #         random_dir = 'backward'
            #     detections = objectDetectionMethod.detect_objects(color_frame)
            #     drive_time += 0.01
                
            for detection in detections:
                coordinates = get_object_wrt_robot(depth_frame, intrinsicsRs, detection)
                robot2rsCam = np.linalg.inv(rsCamToRobot)
                object_pos_wrt_robot = robot2rsCam @ np.array([coordinates['x'], coordinates['y'], coordinates['z'], 1])
                if object_pos_wrt_robot[0] == 0 and object_pos_wrt_robot[2] == 0:
                    print("Abnormality in NN detections x=0, y=0")
                    continue
                else:
                    print(f"object_pos_wrt_robot x: {object_pos_wrt_robot[0]:.2f}, y: {object_pos_wrt_robot[1]:.2f}, z: {object_pos_wrt_robot[2]:.2f}")
                    # Define the function to send the position to control
                def send_position_if_valid(x,y):
                    # x, y = object_pos_wrt_robot

                    # Check if x is in the range of -5 to 5 and y is in the range of 15 to 20
                    if not (-5 <= x <= 5 and y >= 5):
                        control.send_to_XY(x, y)
                    else:
                        print("Position is within the forbidden range, command not sent.")

                    # Example usage
                    # object_pos_wrt_robot = [3, 18]  # Example position
                send_position_if_valid(object_pos_wrt_robot[0], object_pos_wrt_robot[1])

                # control.send_to_XY(object_pos_wrt_robot[0], object_pos_wrt_robot[1])
        except Exception as e:
            print(f"Error in main loop: {e}")
            control.stop_drive()
            break
        except KeyboardInterrupt:
            print("Interrupted by user. Stopping drive.")
            control.stop_drive()
            break

if __name__ == "__main__":
    main()
