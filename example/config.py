import math
class config:
  TAG_POLICY = "HIGHEST_CONFIDENCE"
  TAG_POLICY = "FIRST"
  TAG_DECISION_MARGIN_THRESHOLD = 150
  TAG_SIZE_3IN = 7.62 # cm
  TAG_SIZE_6IN = 15.24 # cm
  GEN_DEBUG_IMAGES = False
  SIZE_OF_CALIBRATION_TAG = 12.7 #cm
  CALIB_PATH = "/home/nvidia/wsp/clawbot/Cortano/calib.txt"
  FIELD = "GAME" 
  FIELD = "HOME" 
  FIELD = "BEDROOM" 
  ROBOT_LENGTH = 45 # cm
  ROBOT_WIDTH = 42 # cm
  ROBOT_HEIGHT = 30 # cm
  ROBOT_RADIUS = math.sqrt(ROBOT_LENGTH**2 + ROBOT_WIDTH**2) / 2
  # FIELD = "GAME"

