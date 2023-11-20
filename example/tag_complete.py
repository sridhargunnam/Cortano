
from datetime import datetime
from io import BytesIO
import multiprocessing as mp
# from multiprocessing import Process, Queue
from pyapriltags import Detector
from scipy.spatial.transform import Rotation as R
from vex_serial import VexControl # Assuming VexControl is a class or a function in vex_serial.py
import object_detection
import camera
import config 
import cv2
import landmarks
import numpy as np
import socket
import json
import struct
import sys
import time
import traceback
import mask_gen as msk
import matplotlib.pyplot as plt
import ball_detection_opencv as ball_detection_opencv
# Global constants

cfg = config.Config()
cfg.TAG_POLICY = "FIRST"
# config.FIELD == "BEDROOM"
cfg.FIELD == "GAME"

RS_CAMERA_QUEUE_SIZE       = 1
DAI_CAMERA_QUEUE_SIZE      = 1
TAG_DETECTION_QUEUE_SIZE   = 1
BALL_DETECTION_QUEUE_SIZE  = 1

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
                          decode_sharpening=1,
                          debug=False)# config.GEN_DEBUG_IMAGES)
    self.camera_params = camera_params
    pass

  def getTagAndPose(self,img, tag_size=cfg.TAG_SIZE_3IN):
    # check if img is color or gray 
    if len(img.shape) == 3:
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
      gray = img
    self.tags = self.at_detector.detect(
      gray, True, self.camera_params[0:4], tag_size)
    if self.tags is not None:
      #get the tag with highest confidence
      max_confidence = cfg.TAG_DECISION_MARGIN_THRESHOLD
      max_confidence_least_pose_err_tag = None
      min_pose_err = cfg.TAG_POSE_ERROR_THRESHOLD
      for tag in self.tags:
        if cfg.TAG_POLICY == "HIGHEST_CONFIDENCE":
          if tag.decision_margin < cfg.TAG_DECISION_MARGIN_THRESHOLD and tag.pose_err > cfg.TAG_POSE_ERROR_THRESHOLD: 
            # print(f'tag.decision_margin = {tag.decision_margin} < {cfg.TAG_DECISION_MARGIN_THRESHOLD}')
            continue
        if tag.decision_margin > max_confidence and tag.pose_err < min_pose_err:
          # print(f'tag.id = {tag.tag_id}, tag.decision_margin = {tag.decision_margin} > {max_confidence} and tag.pose_err = {tag.pose_err} < {min_pose_err}')
          max_confidence = tag.decision_margin
          min_pose_err = tag.pose_err
          max_confidence_least_pose_err_tag = tag
      if max_confidence_least_pose_err_tag is not None:
        # tag_pose = max_confidence_least_pose_err_tag.pose_t
        tag_id = max_confidence_least_pose_err_tag.tag_id
        R = max_confidence_least_pose_err_tag.pose_R
        t = max_confidence_least_pose_err_tag.pose_t
        # make 4 * 4 transformation matrix
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t.ravel()
        return max_confidence_least_pose_err_tag, tag_id, T
    return None, None, None
  


  def getRobotPoseFromTagPose(self, tag_pose, tag_id, Lm2Cam, Cam2Robot):
    if tag_pose is None or tag_id is None or Lm2Cam is None:
      # print(f'getRobotPoseFromTagPose: tag_pose = \n{tag_pose}, tag_id = {tag_id}, Lm2Cam = \n{Lm2Cam}')
      return None
    if tag_id in landmarks.map_apriltag_poses:
      #set numpy print options for precision
      np.set_printoptions(precision=2, suppress=True)
      # print(f'tag_id = {tag_id} in landmarks.map_apriltag_poses')
      Field2Robot = landmarks.map_apriltag_poses[tag_id] @ np.linalg.inv(Lm2Cam) @ Cam2Robot
      print_tag_info = False
      if print_tag_info:
        print(f'landmarks.map_apriltag_poses[{tag_id}] = \n{landmarks.map_apriltag_poses[tag_id]}')
        print("Lm2Cam = \n", Lm2Cam)
        print("Cam2Robot = \n", Cam2Robot)
        print("Field2Robot = \n", Field2Robot)
      # np.savetxt("debug_transformation.txt", np.vstack((Lm2Cam, Cam2Robot)), delimiter=",")
      return Field2Robot
    else:
      return None

def send_command(command, args):
    try:
      print("sending ", command, args)
      client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      client_socket.connect(('localhost', 6000))
      command_data = json.dumps({'command': command, 'args': args})
      client_socket.send(command_data.encode())
      client_socket.close()
    except:
      # print("failed to send", command, args)
      pass

# Step 2: Define camera process functions
def rs_camera_process(camera_params, rs_queue_nn, rs_queue_viz, rs_queue_tag, rs_queue_ocv):
    try:
      # Initialize RealSense camera
      camRS = camera.RealSenseCamera(1280,720) 
      fx, fy, cx, cy, depth_scale = camera_params
      cx.value = camRS.cx
      cy.value = camRS.cy
      fx.value = camRS.fx
      fy.value = camRS.fy
      depth_scale.value = camRS.depth_scale

      # Capture images and put them into the rs_queue
      while True:
        # print("inside rs camera process")
        color, depth = camRS.read()
        if color is None or depth is None:
          continue
        # rotate the color and depthe and put it in the queue
        rs_queue_nn.put(color)
        rs_queue_ocv.put(color)
        rs_queue_viz.put((color, depth))
        # convert to gray for tag detection
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        rs_queue_tag.put(gray)
    except Exception as e:
        print("failed to initialize rs camera", e)
        sys.exit(1)

    

def dai_camera_process(camera_params, dai_queue, dai_queue_viz, dai_queue_tag):
    # Initialize DepthAI camera
    try:
      camDai = camera.DepthAICamera(1920, 1080)
      camera_params_from_device = camDai.getCameraIntrinsics()
      fx, fy, cx, cy = camera_params
      fx.value = camera_params_from_device[0]
      fy.value = camera_params_from_device[1]
      cx.value = camera_params_from_device[2]
      cy.value = camera_params_from_device[3]
    # Capture images and put them into the dai_queue
      while True:
        color, depth = camDai.read()
        if color is None or depth is None:
          continue
        dai_queue.put((color, depth))
        dai_queue_viz.put((color, depth))
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        dai_queue_tag.put(gray)
    except:
      print("failed to initialize dai camera")
      pass



# Step 3: Define detection process functions
def tag_detection_process(camera_queue, tag_detection_queue, robot_state, camera_params, camera_type):
    # AprilTag detection logic
    # Consume images from camera_queue, process them, and put results in tag_detection_queue
  local_config  = config.Config()
  tag_size = local_config.TAG_SIZE_3IN
  try:
    robot_state_local = [0.0, 0.0, 0.0, 0.0]
    camera_params_local = [i.value for i in camera_params][0:4]
    if camera_type.value == 0: # rs camera
      CamToRobot = readCalibrationFile()[0]
      atag = ATag(camera_params_local)
    else: # dai camera
      CamToRobot = readCalibrationFile()[1]
      atag = ATag(camera_params_local)
    while True:
      try :
        # print("inside tag detection process")
        gray = camera_queue.get()
        if gray is None:
          print("didnt get color in tag detection process", camera_type.value)
          continue
        tag, tag_id, tag_pose = atag.getTagAndPose(gray, tag_size)
        Robot2Field = atag.getRobotPoseFromTagPose(tag, tag_id, tag_pose, CamToRobot)
        if Robot2Field is not None:
          robot_state_local[0] = Robot2Field[0, 3]
          robot_state_local[1] = Robot2Field[1, 3]
          robot_state_local[2] = Robot2Field[2, 3]
          robot_state_local[3] = R.from_matrix(Robot2Field[0:3, 0:3]).as_euler('xyz', degrees=True)[2]
          # convert to degrees
          print("Robot x,y, theta = ", robot_state_local[0], robot_state_local[1], robot_state_local[3])
          robot_state.put(robot_state_local)
          tag_detection_queue.put([Robot2Field, tag])
        else:
          tag_detection_queue.put([None, None])
      except Exception as e:
        print("tag detection process failed - inner step", e)
        sys.exit(1)
  except Exception as e:
    print("tag detection process failed", e)
    sys.exit(1)

     

def ball_detection_process(camera_queue, ball_detection_queue):
    # Ball detection logic using ball_detection_opencv
    # Consume images from camera_queue, process them, and put results in ball_detection_queue
    try :
      while True:
          # print("inside ball detection process")
          if not camera_queue.empty():
              color = camera_queue.get()
              if color is None:
                  continue
              detections = object_detection.run_object_detection(color)
              if len(detections) > 0:
                  ball_detection_queue.put(detections)
              else:
                  ball_detection_queue.put([])
    except Exception as e:
        print("ball detection process failed", e)
        sys.exit(1)

class MyQueue:
    def __init__(self, name="noname", max_size=1):
        self.name = name
        self.queue = mp.Queue(max_size)

    def put(self, item):
        if self.queue.full():
            self.queue.get()  # Remove the oldest item
        self.queue.put(item)

    def get(self):
        if self.queue.empty():
            print(f"{self.name} queue is empty")
            # sys.exit(1)        
        return self.queue.get()
    def empty(self):
        return self.queue.empty() 
  


# Step 4: Define the main function
def main():
    ENABLE_DAI = False
    ENABLE_TAG_DETECTION_IN_SEPARATE_PROCESS = True
    # x, y, z, theta # z is supposed to be zero as it's 4 wheel robot
    rs_robot_state    = MyQueue(RS_CAMERA_QUEUE_SIZE)#mp.Array('d', [0.0, 0.0, 0.0, 0.0])
    dai_robot_state   = MyQueue(RS_CAMERA_QUEUE_SIZE)#mp.Array('d', [0.0, 0.0, 0.0, 0.0])
    robot_state_fused =  [0.0, 0.0, 0.0, 0.0]
    command_queue = MyQueue(1)
    calib = np.loadtxt("calib.txt", delimiter=",")
    rsCamToRobot = calib[:4,:]
    daiCamToRobot = calib[4:,:]
    camera_type_rs = mp.Value('i', 0) # 0 for rs, 1 for dai
    camera_type_dai = mp.Value('i', 1) # 0 for rs, 1 for dai



    # Initialize camera parameters
    rs_fx = mp.Value('d', 617.0)
    rs_fy = mp.Value('d', 617.0)
    rs_cx = mp.Value('d', 640.0)
    rs_cy = mp.Value('d', 360.0)
    rs_depth_scale = mp.Value('d', 0.0010000000474974513)
    rs_camera_params = [rs_fx, rs_fy, rs_cx, rs_cy, rs_depth_scale]
    dai_fx = mp.Value('d', 617.0)
    dai_fy = mp.Value('d', 617.0)
    dai_cx = mp.Value('d', 640.0)
    dai_cy = mp.Value('d', 360.0)
    dai_camera_params= [dai_fx, dai_fy, dai_cx, dai_cy]
    # Initialize queues
    rs_queue_nn = MyQueue("rs_queue", RS_CAMERA_QUEUE_SIZE)
    rs_queue_ocv = MyQueue("rs_queue", RS_CAMERA_QUEUE_SIZE)
    rs_queue_tag = MyQueue("rs_queue_tag", RS_CAMERA_QUEUE_SIZE)
    rs_queue_viz = MyQueue("rs_queue_viz", RS_CAMERA_QUEUE_SIZE)
    rs_tag_detection_queue = MyQueue("rs_tag_detection_queue", TAG_DETECTION_QUEUE_SIZE)
    rs_ball_detection_queue = MyQueue("rs_ball_detection_queue", BALL_DETECTION_QUEUE_SIZE)
    rs_ocv_ball_detection_queue = MyQueue("rs_ocv_ball_detection_queue", BALL_DETECTION_QUEUE_SIZE)
    if ENABLE_DAI:
      dai_queue = MyQueue("dai_queue", DAI_CAMERA_QUEUE_SIZE)
      dai_queue_tag = MyQueue("dai_queue_tag", DAI_CAMERA_QUEUE_SIZE)
      dai_queue_viz = MyQueue("dai_queue_viz", RS_CAMERA_QUEUE_SIZE)
      dai_tag_detection_queue = MyQueue("dai_tag_detection_queue", TAG_DETECTION_QUEUE_SIZE)
      dai_ball_detection_queue = MyQueue("dai_ball_detection_queue", BALL_DETECTION_QUEUE_SIZE)

    # Start camera processes
    rs_camera = mp.Process(target=rs_camera_process, args=(rs_camera_params,rs_queue_nn,rs_queue_viz,rs_queue_tag, rs_queue_ocv))
    rs_camera.start()
    

    # Start detection processes
    if ENABLE_TAG_DETECTION_IN_SEPARATE_PROCESS:
      rs_tag_detection = mp.Process(target=tag_detection_process, args=(rs_queue_tag, rs_tag_detection_queue, rs_robot_state, rs_camera_params, camera_type_rs,))
      rs_tag_detection.start()
    else:
      atag = ATag(camera_params=[i.value for i in rs_camera_params][0:4])
    rs_ball_detection = mp.Process(target=ball_detection_process, args=(rs_queue_nn, rs_ball_detection_queue,))
    rs_ball_detection.start()
    rs_ocv_ball_dection = mp.Process(target=ball_detection_opencv.run_object_detection_mp, args=(rs_queue_ocv, rs_ocv_ball_detection_queue, rs_depth_scale.value, ))
    rs_ocv_ball_dection.start()

    if ENABLE_DAI:
      dai_camera = mp.Process(target=dai_camera_process, args=(dai_camera_params, dai_queue, dai_queue_viz,dai_queue_tag))
      dai_camera.start()
      dai_tag_detection = mp.Process(target=tag_detection_process, args=(dai_queue_tag, dai_tag_detection_queue, dai_robot_state, dai_camera_params, camera_type_dai,))
      dai_ball_detection = mp.Process(target=ball_detection_process, args=(dai_queue, dai_ball_detection_queue,))
      dai_tag_detection.start()
      dai_ball_detection.start()
    SIZE_OF_TENNIS_BALL = 6.54 # centimeters

 
    if cfg.FIELD == "GAME":
      tag_size = cfg.TAG_SIZE_3IN

    # Main loop for processing data
    while True:
        try:
            # Retrieve data from detection queues
            if rs_ocv_ball_detection_queue.empty():
               pass
            if rs_ball_detection_queue.empty():
               pass
              #  print("rs ball detection is empty")
            if rs_queue_viz.empty():
                # print("rs queue viz is empty")
                # time.sleep(0.1)
                continue
            colorRS, depthRS = rs_queue_viz.get()
            def visualize_ball_detection(color, depth, ball_detections, ball_color):
              for detection in ball_detections:
                  center_x = detection['center_x']
                  center_y = detection['center_y']
                  width = detection['width']
                  height = detection['height']
                  radius = (width + height) / 4
                  center = (int(center_x), int(center_y))
                  cv2.circle(color, center, int(radius), ball_color, 2)
                  depth = depth[int(center_y)][int(center_x)]
                  depth = depth * 0.0010000000474974513
            rs_ball_detections = rs_ball_detection_queue.get()
            rs_ocv_ball_detections = rs_ocv_ball_detection_queue.get()
            color_green = (0, 255, 0) # green
            color_red = (0, 0, 255) # red
            if colorRS is None or depthRS is None:
              continue
            elif colorRS.shape == 3 and depthRS.shape == 2:
              visualize_ball_detection(colorRS, depthRS, rs_ball_detections, color_green)
              visualize_ball_detection(colorRS, depthRS, rs_ocv_ball_detections, color_red)
            else:
              print("colorRS shape = ", colorRS.shape)
              print("depthRS shape = ", depthRS.shape)

              
            if ENABLE_TAG_DETECTION_IN_SEPARATE_PROCESS:
              if not rs_tag_detection_queue.empty():
                Robot2Field, tag = rs_tag_detection_queue.get()
                if Robot2Field is not None:
                  robot_state_fused[0] = Robot2Field[0, 3]
                  robot_state_fused[1] = Robot2Field[1, 3]
                  robot_state_fused[2] = Robot2Field[2, 3]
                  robot_state_fused[3] = R.from_matrix(Robot2Field[0:3, 0:3]).as_euler('xyz', degrees=True)[2]
                  print("robots x, y, theta are ", robot_state_fused[0], robot_state_fused[1], robot_state_fused[3])
                  # visualize the tag on the image
                  if tag is not None:
                    for idx in range(len(tag.corners)):
                      cv2.line(colorRS, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))
                    cv2.putText(colorRS, str(tag.tag_id),
                                org=(tag.center[0].astype(int), tag.center[1].astype(int)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.8,
                                color=(0, 0, 255))
                
            else: # ENABLE_TAG_DETECTION_IN_SEPARATE_PROCESS:
              tag, tag_id, Lm2Cam = atag.getTagAndPose(colorRS, tag_size)
              print("tag_id = ", tag_id)
              Robot2Field = atag.getRobotPoseFromTagPose(tag, tag_id, Lm2Cam, rsCamToRobot)
              # print("Robot2Field = ", Robot2Field)
              if Robot2Field is not None:
                robot_state_fused[0] = Robot2Field[0, 3]
                robot_state_fused[1] = Robot2Field[1, 3]
                robot_state_fused[2] = Robot2Field[2, 3]
                robot_state_fused[3] = R.from_matrix(Robot2Field[0:3, 0:3]).as_euler('xyz', degrees=True)[2]
                print("robots x, y, theta are ", robot_state_fused[0], robot_state_fused[1], robot_state_fused[3])
            if ENABLE_DAI:
              dai_ball_detections = dai_ball_detection_queue.get()
              colorDAI, depthDAI = dai_queue_viz.get()

              for result in dai_ball_detections:
                  center_x = result['center_x']
                  center_y = result['center_y']
                  width = result['width']
                  height = result['height']
                  radius = (width + height) / 4
                  center = (int(center_x), int(center_y))
                  cv2.circle(colorDAI, center, int(radius), (0, 255, 0), 2)
                  cx = dai_camera_params[0].value
                  cy = dai_camera_params[1].value
                  fx = dai_camera_params[2].value
                  fy = dai_camera_params[3].value
                  z = 0.5 * SIZE_OF_TENNIS_BALL * fx / radius
                  x = (center_x - cx) * z / fx
                  y = (center_y - cy) * z / fy

              #stack images
              resize_scale = colorRS.shape[0] / colorDAI.shape[0]
              resized_colorDAI = cv2.resize(colorDAI, (int(colorDAI.shape[1] * resize_scale), colorRS.shape[0]))

              color_combined = np.hstack((colorRS, resized_colorDAI))

            #downscale = 0.25 to view
            # cv2.imshow('color_combined', cv2.resize(color_combined, (0,0), fx=0.25, fy=0.25))
            # cv2.imshow('color_combined', color_combined)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # else:
            # downscale and rotate 90 degrees for viewing
            downscale = 0.25
            colorRS = cv2.rotate(colorRS, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow('colorRS', cv2.resize(colorRS, (0,0), fx=downscale, fy=downscale))

            # cv2.imshow('colorRS', colorRS)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # Process data for visualization and VEX control commands
            # pass
        except KeyboardInterrupt:
            break
        except Exception as e:
            print('Error:', e)
            traceback.print_exc()
            # sys.exit(1)
            time.sleep(1)


    # Cleanup and close processes
    rs_camera.terminate()
    if ENABLE_TAG_DETECTION_IN_SEPARATE_PROCESS:
      rs_tag_detection.terminate()
    rs_ball_detection.terminate()
    rs_ocv_ball_dection.terminate()
    if ENABLE_DAI:
      dai_camera.terminate()
      dai_tag_detection.terminate()
      dai_ball_detection.terminate()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
