
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

config = config.Config()
config.TAG_POLICY = "FIRST"
# config.FIELD == "BEDROOM"
config.FIELD == "GAME"

RS_CAMERA_QUEUE_SIZE = 100
DAI_CAMERA_QUEUE_SIZE = 100
TAG_DETECTION_QUEUE_SIZE = 100
BALL_DETECTION_QUEUE_SIZE = 100

def readCalibrationFile(path=config.CALIB_PATH):
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

  def getTagAndPose(self,color, tag_size=config.TAG_SIZE_3IN):
    self.tags = self.at_detector.detect(
      cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), True, self.camera_params[0:4], tag_size)
    if self.tags is not None:
      #get the tag with highest confidence
      max_confidence = 0
      max_confidence_tag = None
      for tag in self.tags:
        if config.TAG_POLICY == "HIGHEST_CONFIDENCE":
          if tag.decision_margin < config.TAG_DECISION_MARGIN_THRESHOLD: 
            print(f'tag.decision_margin = {tag.decision_margin} < {config.TAG_DECISION_MARGIN_THRESHOLD}')
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
def rs_camera_process(camera_params, rs_queue, rs_queue_viz):
    # Initialize RealSense camera
    camRS = camera.RealSenseCamera(1280,720) 
    cx, cy, fx, fy, depth_scale = camera_params
    cx.value = camRS.cx
    cy.value = camRS.cy
    fx.value = camRS.fx
    fy.value = camRS.fy
    depth_scale.value = 0.001 #camRS.depth_scale

    # Capture images and put them into the rs_queue
    while True:
      color, depth = camRS.read()
      if color is None or depth is None:
        continue
      rs_queue.put((color, depth))
      rs_queue_viz.put((color, depth))
    

def dai_camera_process(dai_queue):
    # Initialize DepthAI camera
    # Capture images and put them into the dai_queue
    pass

# Step 3: Define detection process functions
def tag_detection_process(camera_queue, tag_detection_queue):
    # AprilTag detection logic
    # Consume images from camera_queue, process them, and put results in tag_detection_queue
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
    command_queue = mp.Queue()
    calib = np.loadtxt("calib.txt", delimiter=",")
    rsCamToRobot = calib[:4,:]
    # daiCamToRobot = calib[4:,:]

    # Initialize camera parameters
    cx = mp.Value('d', 640.0)
    cy = mp.Value('d', 360.0)
    fx = mp.Value('d', 617.0)
    fy = mp.Value('d', 617.0)
    depth_scale = mp.Value('d', 0.001)
    camera_params = [cx, cy, fx, fy, depth_scale]
    # Initialize queues
    rs_queue = mp.Queue(RS_CAMERA_QUEUE_SIZE)
    rs_queue_viz = mp.Queue(RS_CAMERA_QUEUE_SIZE)
    # dai_queue = mp.Queue(DAI_CAMERA_QUEUE_SIZE)
    rs_tag_detection_queue = mp.Queue(TAG_DETECTION_QUEUE_SIZE)
    # dai_tag_detection_queue = mp.Queue(TAG_DETECTION_QUEUE_SIZE)
    rs_ball_detection_queue = mp.Queue(BALL_DETECTION_QUEUE_SIZE)
    # dai_ball_detection_queue = mp.Queue(BALL_DETECTION_QUEUE_SIZE)

    # Start camera processes
    rs_camera = mp.Process(target=rs_camera_process, args=(camera_params,rs_queue,rs_queue_viz,))
    # dai_camera = mp.Process(target=dai_camera_process, args=(dai_queue,))
    rs_camera.start()
    # dai_camera.start()

    # Start detection processes
    # rs_tag_detection = mp.Process(target=tag_detection_process, args=(rs_queue, rs_tag_detection_queue))
    # dai_tag_detection = mp.Process(target=tag_detection_process, args=(dai_queue, dai_tag_detection_queue))
    rs_ball_detection = mp.Process(target=ball_detection_process, args=(rs_queue, rs_ball_detection_queue))
    # dai_ball_detection = mp.Process(target=ball_detection_process, args=(dai_queue, dai_ball_detection_queue))
    # rs_tag_detection.start()
    # dai_tag_detection.start()
    rs_ball_detection.start()
    # dai_ball_detection.start()

    # Main loop for processing data
    while True:
        print("in main loop")
        try:
            # Retrieve data from detection queues
            rs_ball_detections = rs_ball_detection_queue.get()
            colorRS, depthRS = rs_queue_viz.get()
            for result in rs_ball_detections:
                center_x = result['center_x']
                center_y = result['center_y']
                width = result['width']
                height = result['height']
                radius = (width + height) / 4
                center = (int(center_x), int(center_y))
                cv2.circle(colorRS, center, int(radius), (0, 255, 0), 2)
  
                depth_ = depthRS[int(center_y)][int(center_x)]
                depth_ = depth_ * depth_scale.value
                print("In main depth_scale = ", depth_scale.value)
            cv2.imshow('RealSense', colorRS)
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
    # dai_camera.terminate()
    # rs_tag_detection.terminate()
    # dai_tag_detection.terminate()
    rs_ball_detection.terminate()
    # dai_ball_detection.terminate()

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
