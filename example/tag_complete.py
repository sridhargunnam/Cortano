
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
# Global constants

cfg = config.Config()
cfg.TAG_POLICY = "FIRST"
# config.FIELD == "BEDROOM"
cfg.FIELD == "GAME"

RS_CAMERA_QUEUE_SIZE       = 10
DAI_CAMERA_QUEUE_SIZE      = 10
TAG_DETECTION_QUEUE_SIZE   = 10
BALL_DETECTION_QUEUE_SIZE  = 10

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

  def getTagAndPose(self,color, tag_size=cfg.TAG_SIZE_3IN):
    self.tags = self.at_detector.detect(
      cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), True, self.camera_params[0:4], tag_size)
    if self.tags is not None:
      #get the tag with highest confidence
      max_confidence = 0
      max_confidence_tag = None
      for tag in self.tags:
        if cfg.TAG_POLICY == "HIGHEST_CONFIDENCE":
          if tag.decision_margin < cfg.TAG_DECISION_MARGIN_THRESHOLD: 
            print(f'tag.decision_margin = {tag.decision_margin} < {cfg.TAG_DECISION_MARGIN_THRESHOLD}')
            continue
        if tag.decision_margin > max_confidence:
          max_confidence = tag.decision_margin
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
      #set numpy print options for precision
      np.set_printoptions(precision=2, suppress=True)
      # print(f'tag_id = {tag_id} in landmarks.map_apriltag_poses')
      Field2Robot = landmarks.map_apriltag_poses[tag_id] @ np.linalg.inv(Lm2Cam) @ Cam2Robot
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
def rs_camera_process(camera_params, rs_queue, rs_queue_viz, rs_queue_tag):
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
      color, depth = camRS.read()
      if color is None or depth is None:
        continue
      rs_queue.put((color, depth))
      rs_queue_viz.put((color, depth))
      # rs_queue_tag.put((color, depth))

    

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
        dai_queue_tag.put((color, depth))
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
    with camera_params.get_lock():
      camera_params_local = [i.value for i in camera_params]
    if camera_type.value == 0: # rs camera
      CamToRobot = readCalibrationFile()[0]
      atag = ATag(camera_params_local)
    else: # dai camera
      CamToRobot = readCalibrationFile()[1]
      atag = ATag(camera_params_local)
    while True:
      color, _ = camera_queue.get()
      if color is None:
        print("didnt get color in tag detection process", camera_type.value)
        continue
      tag, tag_id, tag_pose = atag.getTagAndPose(color, tag_size)
      Robot2Field = atag.getRobotPoseFromTagPose(tag, tag_id, tag_pose, CamToRobot)
      if Robot2Field is not None:
        robot_state_local[0] = Robot2Field[0, 3]
        robot_state_local[1] = Robot2Field[1, 3]
        robot_state_local[2] = Robot2Field[2, 3]
        robot_state_local[3] = R.from_matrix(Robot2Field[0:3, 0:3]).as_euler('xyz', degrees=True)[2]
        robot_state.put(robot_state_local)
        tag_detection_queue.put(Robot2Field)
      # else:
        # tag_detection_queue.put(None)
  except:
    print("failed to initialize/run tag detection process")
    pass

def ball_detection_process(camera_queue, ball_detection_queue):
    # Ball detection logic using ball_detection_opencv
    # Consume images from camera_queue, process them, and put results in ball_detection_queue
    while True:
        if not camera_queue.empty():
            color, depth = camera_queue.get()
            if color is None or depth is None:
                continue
            detections = object_detection.run_object_detection(color)
            if len(detections) > 0:
                ball_detection_queue.put(detections)
            else:
                ball_detection_queue.put([])



# Step 4: Define the main function
def main():
    ENABLE_DAI = False
    # x, y, z, theta # z is supposed to be zero as it's 4 wheel robot
    rs_robot_state    = mp.Queue(RS_CAMERA_QUEUE_SIZE)#mp.Array('d', [0.0, 0.0, 0.0, 0.0])
    dai_robot_state   = mp.Queue(RS_CAMERA_QUEUE_SIZE)#mp.Array('d', [0.0, 0.0, 0.0, 0.0])
    robot_state_fused =  [0.0, 0.0, 0.0, 0.0]
    command_queue = mp.Queue()
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
    rs_depth_scale = mp.Value('d', 0.001)
    rs_camera_params = [rs_fx, rs_fy, rs_cx, rs_cy, rs_depth_scale]
    dai_fx = mp.Value('d', 617.0)
    dai_fy = mp.Value('d', 617.0)
    dai_cx = mp.Value('d', 640.0)
    dai_cy = mp.Value('d', 360.0)
    dai_camera_params= [dai_fx, dai_fy, dai_cx, dai_cy]
    # Initialize queues
    rs_queue = mp.Queue(RS_CAMERA_QUEUE_SIZE)
    rs_queue_tag = mp.Queue(RS_CAMERA_QUEUE_SIZE)
    rs_queue_viz = mp.Queue(RS_CAMERA_QUEUE_SIZE)
    # rs_tag_detection_queue = mp.Queue(TAG_DETECTION_QUEUE_SIZE)
    rs_ball_detection_queue = mp.Queue(BALL_DETECTION_QUEUE_SIZE)
    if ENABLE_DAI:
      dai_queue = mp.Queue(DAI_CAMERA_QUEUE_SIZE)
      dai_queue_tag = mp.Queue(DAI_CAMERA_QUEUE_SIZE)
      dai_queue_viz = mp.Queue(RS_CAMERA_QUEUE_SIZE)
      dai_tag_detection_queue = mp.Queue(TAG_DETECTION_QUEUE_SIZE)
      dai_ball_detection_queue = mp.Queue(BALL_DETECTION_QUEUE_SIZE)

    # Start camera processes
    rs_camera = mp.Process(target=rs_camera_process, args=(rs_camera_params,rs_queue,rs_queue_viz,rs_queue_tag))
    rs_camera.start()
    

    # Start detection processes
    # rs_tag_detection = mp.Process(target=tag_detection_process, args=(rs_queue_tag, rs_tag_detection_queue, rs_robot_state, rs_camera_params, camera_type_rs,))
    rs_ball_detection = mp.Process(target=ball_detection_process, args=(rs_queue, rs_ball_detection_queue,))
    # rs_tag_detection.start()
    rs_ball_detection.start()

    if ENABLE_DAI:
      dai_camera = mp.Process(target=dai_camera_process, args=(dai_camera_params, dai_queue, dai_queue_viz,dai_queue_tag))
      dai_camera.start()
      dai_tag_detection = mp.Process(target=tag_detection_process, args=(dai_queue_tag, dai_tag_detection_queue, dai_robot_state, dai_camera_params, camera_type_dai,))
      dai_ball_detection = mp.Process(target=ball_detection_process, args=(dai_queue, dai_ball_detection_queue,))
      dai_tag_detection.start()
      dai_ball_detection.start()
    SIZE_OF_TENNIS_BALL = 6.54 # centimeters

    atag = ATag(camera_params=[i.value for i in rs_camera_params][0:4])
    if cfg.FIELD == "GAME":
      tag_size = cfg.TAG_SIZE_3IN

    # Main loop for processing data
    while True:
        try:
            # Retrieve data from detection queues
            if rs_ball_detection_queue.empty():
               pass
              #  print("rs ball detection is empty")
            if rs_queue_viz.empty():
                # print("rs queue viz is empty")
                time.sleep(0.1)
                continue
            rs_ball_detections = rs_ball_detection_queue.get()
            colorRS, depthRS = rs_queue_viz.get()
            tag, tag_id, Lm2Cam = atag.getTagAndPose(colorRS, tag_size)
            # print("tag_id = ", tag_id)
            Robot2Field = atag.getRobotPoseFromTagPose(tag, tag_id, Lm2Cam, rsCamToRobot)
            # print("Robot2Field = ", Robot2Field)
            if Robot2Field is not None:
              robot_state_fused[0] = Robot2Field[0, 3]
              robot_state_fused[1] = Robot2Field[1, 3]
              robot_state_fused[2] = Robot2Field[2, 3]
              robot_state_fused[3] = R.from_matrix(Robot2Field[0:3, 0:3]).as_euler('xyz', degrees=True)[2]
              # print("robot_state_fused = ", robot_state_fused)
            for result in rs_ball_detections:
                center_x = result['center_x']
                center_y = result['center_y']
                width = result['width']
                height = result['height']
                radius = (width + height) / 4
                center = (int(center_x), int(center_y))
                cv2.circle(colorRS, center, int(radius), (0, 255, 0), 2)
                depth_ = depthRS[int(center_y)][int(center_x)]
                depth_ = depth_ * rs_depth_scale.value
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

              # cv2.imshow('color_combined', color_combined)
              # if cv2.waitKey(1) & 0xFF == ord('q'):
              #     break
            else:
              cv2.imshow('colorRS', colorRS)
              if cv2.waitKey(1) & 0xFF == ord('q'):
                  break
            # Process data for visualization and VEX control commands
            pass
        except KeyboardInterrupt:
            break
        except Exception as e:
            print('Error:', e)
            traceback.print_exc()
            time.sleep(1)

    # Cleanup and close processes
    rs_camera.terminate()
    # rs_tag_detection.terminate()
    rs_ball_detection.terminate()
    if ENABLE_DAI:
      dai_camera.terminate()
      dai_tag_detection.terminate()
      dai_ball_detection.terminate()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
