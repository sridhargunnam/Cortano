import math
class Config:
  TAG_POLICY = "HIGHEST_CONFIDENCE"
  TAG_POLICY = "FIRST"
  TAG_DECISION_MARGIN_THRESHOLD = 150
  TAG_POSE_ERROR_THRESHOLD = 0.01
  TAG_SIZE_3IN = 7.62 # cm
  TAG_SIZE_6IN = 15.24 # cm
  GEN_DEBUG_IMAGES = False
  SIZE_OF_CALIBRATION_TAG = 12.7 #cm
  CALIB_PATH = "/home/nvidia/wsp/clawbot/Cortano/calib.txt"
  # FIELD = "GAME" 
  FIELD = "HOME" 
  # FIELD = "BEDROOM" 
  ROBOT_LENGTH = 45 # cm
  ROBOT_WIDTH = 42 # cm
  ROBOT_HEIGHT = 30 # cm
  ROBOT_RADIUS = math.sqrt(ROBOT_LENGTH**2 + ROBOT_WIDTH**2) / 2
  WHEEL_RADIUS = 5.08 # in cm , 2 inch
  WHEEL_CIRCUMFERENCE = 2 * math.pi * WHEEL_RADIUS
  WHEEL_DISTANCE = 29 # cm
  WHEEL_DISTANCE_SIDE = 21 # cm
  
  
  # FIELD = "GAME"
  FIELD_WIDTH = 366 # cm
  FIELD_HEIGHT = 366 # cm
  CLAW_LENGTH = 14 # cm
  CREATE_MASK = False
  MASK_PATH = "/home/nvidia/wsp/clawbot/Cortano/mask.png"

  LIVE_BALL_DEBUG = True

