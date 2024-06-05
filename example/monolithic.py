import cv2
import numpy as np
import socket
import json
import camera
import vex_serial as vex
import landmarks


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

from pyapriltags import Detector
import config 
cfg = config.Config()

def readCalibrationFile(path=cfg.CALIB_PATH):
  calib = np.loadtxt(path, delimiter=",")
  rsCamToRobot = calib[:4,:]
  daiCamToRobot = calib[4:,:]
  return rsCamToRobot, daiCamToRobot

class ATag:
  def __init__(self, camera_params) -> None:
    self.at_detector = Detector(families='tag16h5',
                          nthreads=1,
                          quad_decimate=1.0,
                          quad_sigma=0.0,
                          refine_edges=1,
                          decode_sharpening=1.2,
                          debug=False)# cfg.GEN_DEBUG_IMAGES)
    self.camera_params = camera_params
    pass

  def getTagAndPose(self,color, tag_size=cfg.TAG_SIZE_3IN):
    self.tags = self.at_detector.detect(
      cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), True, self.camera_params[0:4], tag_size)
    if self.tags is not None:
      #get the tag with highest confidence
      max_confidence = cfg.TAG_DECISION_MARGIN_THRESHOLD
      min_pose_err = cfg.TAG_POSE_ERROR_THRESHOLD
      max_confidence_tag = None
      for tag in self.tags:
        # print all tag information like pose err and decision margin
        # if tag.tag_id == 6:
          # print(f'tag.decision_margin = {tag.decision_margin}, tag.pose_err = {tag.pose_err}, tag.tag_id = {tag.tag_id}')
          # print(tag.pose_t)
        
        if cfg.TAG_POLICY == "HIGHEST_CONFIDENCE":
          if tag.decision_margin < cfg.TAG_DECISION_MARGIN_THRESHOLD and tag.pose_err > cfg.TAG_POSE_ERROR_THRESHOLD:
            # print(f'tag.decision_margin = {tag.decision_margin} < {cfg.TAG_DECISION_MARGIN_THRESHOLD}')
            continue
        if tag.decision_margin > max_confidence and tag.pose_err < min_pose_err:
          max_confidence = tag.decision_margin
          min_pose_err = tag.pose_err
          max_confidence_tag = tag
      if max_confidence_tag is not None:
        # tag_pose = max_confidence_tag.pose_t
        tag_id = max_confidence_tag.tag_id
        R = max_confidence_tag.pose_R
        t = max_confidence_tag.pose_t
        # make 4 * 4 transformation matrix
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t.ravel()
        return max_confidence_tag, tag_id, T
    return None, None, None
  
  def getRobotPoseFromTagPose(self, tag_pose, tag_id, Lm2Cam, Cam2Robot):
    if tag_pose is None or tag_id is None or Lm2Cam is None:
      # print(f'getRobotPoseFromTagPose: tag_pose = \n{tag_pose}, tag_id = {tag_id}, Lm2Cam = \n{Lm2Cam}')
      return None
    if tag_id in landmarks.map_apriltag_poses:
      # print(f'tag_id = {tag_id} in landmarks.map_apriltag_poses')
      Field2Robot = landmarks.map_apriltag_poses[tag_id] @ np.linalg.inv(Lm2Cam) @ Cam2Robot
      # print(f'landmarks.map_apriltag_poses[{tag_id}] = \n{landmarks.map_apriltag_poses[tag_id]}')
      # print("Lm2Cam = \n", Lm2Cam)
      # print("Cam2Robot = \n", Cam2Robot)
      # print("Field2Robot = \n", Field2Robot)
    # np.savetxt("debug_transformation.txt", np.vstack((Lm2Cam, Cam2Robot)), delimiter=",")
      return Field2Robot
    else:
      return None

ROBOT_STATE = {"OBJECT_SCAN", "OBJECT_DETECTED", "REACHED_OBJECT_GRAB_LOC", "GRABBED_OBJ", "TARGET_TAG_FOUND", "MOVE_TO_TARGET", "DROP_BALL", "SAFE_LOCATION_TO_SCAN", "STUCK"}


def main():
    camRS = camera.RealSenseCamera(640, 360)
    camDAI = camera.DepthAICamera(640, 360)
    objectDetectionMethod = ObjectDetectionML()
    objectDetectionMethodAlternate = ObjectDetectionOpenCV()
    enable_opencv_when_ML_is_not_available = False 
    calib_data = camera.read_calibration_data()
    intrinsicsRs = calib_data['intrinsicsRs']
    rsCamToRobot = calib_data['rsCamToRobot']
    daiCamToRobot = calib_data['daiCamToRobot']
    robot = vex.VexCortex("/dev/ttyUSB0")
    control = vex.VexControl(robot)
    control.stop_drive()
    control.claw([20, "open", 1, 0.8])
    control.claw([20, "open", 1, 0.8])
    prev_object_pos_wrt_robot = None
    random_dir = 'backward'
    drive_time = 0.05
    entering_ball_hold_state = False
    entered_goal_state = False
    orient_towards_goal = False


    # robot_state = ROBOT_STATE

    #
    camera_params = [ 
        intrinsicsRs['fx'],
        intrinsicsRs['fy'],
        intrinsicsRs['cx'],
        intrinsicsRs['cy'],
        ]
    atag = ATag(camera_params)
    # detect the tag and get the pose
    if cfg.FIELD == "HOME" or cfg.FIELD == "GAME":
        tag_size = cfg.TAG_SIZE_3IN # centimeters
    elif cfg.FIELD == "BEDROOM":
        tag_size = cfg.TAG_SIZE_6IN

    grab_phase = True
    
    while True:
        drive_time = 0.05
        color_frame_RS, depth_frame_RS = camRS.read()
        color_frame_DAI, depth_frame_DAI = camDAI.read()
        # camDAI.read()
        if color_frame_RS is None or depth_frame_RS is None:
            print('color_frame or depth frame is none in RS')
            continue
        if color_frame_RS is None or depth_frame_RS is None:
            print('color_frame or depth frame is none in DAI')
            continue
        if grab_phase:
            try:
                detections = objectDetectionMethod.detect_objects(color_frame_RS)
                if len(detections) == 0 and enable_opencv_when_ML_is_not_available:
                    detections = objectDetectionMethodAlternate.detect_objects(color_frame_RS)
                # if len(detections) == 0:
                #     print("No detections, rotating clockwise.")
                #     control.rotateRobot([0.05, vex.ROTATION_DIRECTION["clockwise"], vex.MINIMUM_INPLACE_ROTATION_SPEED])
                #     detections = objectDetectionMethod.detect_objects(color_frame_RS)

                # if len(detections) == 0:
                #     print("No detections, rotating counterclockwise.")
                #     control.rotateRobot([0.05, vex.ROTATION_DIRECTION["counter_clockwise"], vex.MINIMUM_INPLACE_ROTATION_SPEED])
                #     detections = objectDetectionMethod.detect_objects(color_frame_RS)

                # if len(detections) == 0:
                #     print("No detections, moving forward.")
                #     control.drive(['forward', 30, 0.7])
                #     detections = objectDetectionMethod.detect_objects(color_frame_RS)

                # if len(detections) == 0:
                #     print("No detections, moving backward.")
                #     control.drive(['backward', 30, 0.7])
                #     detections = objectDetectionMethod.detect_objects(color_frame_RS)

                while len(detections) == 0:
                    print("No detections, rotating clockwise until we find an object.")
                    control.rotateRobot([0.06, vex.ROTATION_DIRECTION["clockwise"], vex.MINIMUM_INPLACE_ROTATION_SPEED])
                    import time
                    time.sleep(0.03)
                    color_frame_RS, depth_frame_RS = camRS.read()
                    color_frame_DAI, depth_frame_DAI = camDAI.read()
                    # while
                    detections = objectDetectionMethod.detect_objects(color_frame_RS)
                    
                    
                for detection in detections:
                    coordinates = get_object_wrt_robot(depth_frame_RS, intrinsicsRs, detection)
                    robot2rsCam = np.linalg.inv(rsCamToRobot)
                    object_pos_wrt_robot = robot2rsCam @ np.array([coordinates['x'], coordinates['y'], coordinates['z'], 1])
                    prev_object_pos_wrt_robot = object_pos_wrt_robot
                    if object_pos_wrt_robot[0] == 0 and object_pos_wrt_robot[2] == 0:
                        print("Abnormality in NN detections x=0, y=0")
                        continue
                    else:
                        print(f"object_pos_wrt_robot x: {object_pos_wrt_robot[0]:.2f}, y: {object_pos_wrt_robot[1]:.2f}, z: {object_pos_wrt_robot[2]:.2f}")
                        # Define the function to send the position to control
                    def send_position_if_valid(x,y, entering_ball_hold_state, grab_phase):
                        # x, y = object_pos_wrt_robot
                        # Check if x is in the range of -5 to 5 and y is in the range of 15 to 20
                        if not (-5 <= x <= 5 and y <= 15):
                            if entering_ball_hold_state:
                                entering_ball_hold_state = False
                                control.claw([20, "open", 1, 0.8])
                                control.claw([20, "open", 1, 0.8])
                            control.send_to_XY(x, y)
                        elif (-5 <= x <= 5 and y <= 15):
                            entering_ball_hold_state = True
                            print("Position is within the grab range. Initiating grab command")
                            control.drive(['forward', 35, 1.3])
                            control.rotateRobot([0.02, vex.ROTATION_DIRECTION["clockwise"], vex.MINIMUM_INPLACE_ROTATION_SPEED])
                            control.rotateRobot([0.02, vex.ROTATION_DIRECTION["clockwise"], vex.MINIMUM_INPLACE_ROTATION_SPEED])
                            control.claw([20, "open", 1, 0.8])
                            control.claw([20, "open", 1, 0.8])
                            control.claw([20, "close", 1, 0.8])
                            control.claw([20, "close", 1, 0.8])
                            control.claw([20, "close", 1, 0.8])
                            control.drive(['backward', 30, 0.7])
                            if control.robot.sensors()[2] == 1:
                                print("Hurray!! ball held detected")
                            grab_phase = False
                        return grab_phase, entering_ball_hold_state
                        # # armPosition=ARM_POSITION.low, motor=2, error=20
                        # send_command('update_robot_move_arm', ['high', 2, 20])
                        # # send_command('update_robot_move_arm', ['low', 2, 20])
                        # send_command("stop_drive", None)

                        # Example usage
                        # object_pos_wrt_robot = [3, 18]  # Example position
                    grab_phase, entering_ball_hold_state = send_position_if_valid(object_pos_wrt_robot[0], object_pos_wrt_robot[1], entering_ball_hold_state, grab_phase)
                    orient_towards_goal = False

                    # control.send_to_XY(object_pos_wrt_robot[0], object_pos_wrt_robot[1])
            except Exception as e:
                print(f"Error in main loop: {e}")
                control.stop_drive()
                break
            except KeyboardInterrupt:
                print("Interrupted by user. Stopping drive.")
                control.stop_drive()
                break
        else:
            # tagRS, tag_idRS, Lm2CamRS = atag.getTagAndPose(color_frame_RS, tag_size)
            tagDAI, tag_idDAI, Lm2CamDAI = atag.getTagAndPose(color_frame_DAI, tag_size)
            while tagDAI is None and not orient_towards_goal:
                    print("No detections, rotating clockwise until we find an landmark.")
                    # while
                    control.rotateRobot([0.06, vex.ROTATION_DIRECTION["clockwise"], vex.MINIMUM_INPLACE_ROTATION_SPEED])
                    import time
                    time.sleep(0.03)
                    # color_frame_RS, depth_frame_RS = camRS.read()
                    color_frame_DAI, depth_frame_DAI = camDAI.read()
                    # detections = objectDetectionMethod.detect_objects(color_frame_RS)
                    # tagRS, tag_idRS, Lm2CamRS = atag.getTagAndPose(color_frame_RS, tag_size)
                    tagDAI, tag_idDAI, Lm2CamDAI = atag.getTagAndPose(color_frame_DAI, tag_size)
                    if tagDAI is not None:
                        orient_towards_goal = True

            goal_position = [0,0]
            angle_x_degrees = 0
            # if tagRS is not None:
            #     Robot2FieldRS = atag.getRobotPoseFromTagPose(tagRS, tag_idRS, Lm2CamRS, rsCamToRobot)
            #     if Robot2FieldRS is not None:
            #         print("tagRS  = ", tag_idRS)
            #         print("Robot2FieldRS = ", Robot2FieldRS[0:3,3])
            #         goal_position = -Robot2FieldRS[0:3,3][0] , Robot2FieldRS[0:3,3][1]

            if tagDAI is not None:
                Robot2FieldDAI = atag.getRobotPoseFromTagPose(tagDAI, tag_idDAI, Lm2CamDAI, daiCamToRobot)
                if Robot2FieldDAI is not None:
                    print("tagDAI  = ", tag_idDAI)
                    print("Robot2FieldDAI = ", Robot2FieldDAI[0:3,3])
                    goal_position = Robot2FieldDAI[0:3,3][0] , -Robot2FieldDAI[0:3,3][1]
                    angle_x = np.arctan2(goal_position[0], goal_position[1])
                    # Convert angle to degrees
                    angle_x_degrees = np.degrees(angle_x)

                def send_position_if_valid_goal(x,y, angle_x_degrees, entered_goal_state):
                    # x, y = object_pos_wrt_robot
                    # Check if x is in the range of -5 to 5 and y is in the range of 15 to 20
                    if not (abs(y) <= 20):
                        control.send_to_XY_Theta_simple(x, abs(y), angle_x_degrees)
                    else:
                        if not entered_goal_state:
                            entered_goal_state = True
                            control.claw([20, "open", 1, 0.8])
                            control.claw([20, "open", 1, 0.8])
                            return entered_goal_state
                    return entered_goal_state
                entered_goal_state = send_position_if_valid_goal(goal_position[0], goal_position[1], angle_x_degrees, entered_goal_state)
                if entered_goal_state:
                    entered_goal_state = False
                    entering_ball_hold_state = False
                    grab_phase = True


                    # print(Robot2FieldRS[:3,0], Robot2FieldRS[:3,1], Robot2FieldRS[:3,2])
                # print("tag_idRS = ", tag_idRS)
                # print("Lm2CamRS ", Lm2CamRS)
                # np.linalg.inv(Lm2Cam) @ Cam2Robot
            

if __name__ == "__main__":
    main()
