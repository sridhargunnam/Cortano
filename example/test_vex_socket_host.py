# this file can be delteted. It is just for testing the apriltag detection

from datetime import datetime
from io import BytesIO
from multiprocessing import Process, Queue
from pyapriltags import Detector
from scipy.spatial.transform import Rotation as R
import camera
import config 
import cv2
import landmarks
import numpy as np
import socket
import json
import struct
import sys
import vex_serial as vex

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

def main():
    #    control.claw(20, clawAction.Close, drive_time=1.5)
    send_command('claw', {'claw_action': 'close', 'drive_time': 1.5})
    send_command('claw', {'claw_action': 'open', 'drive_time': 1.5})

    # control.update_robot_move_arm(armPosition=ARM_POSITION.low)
    # send_command('update_robot_move_arm', {'arm_position': 'low'})
    # send_command('update_robot_move_arm', {'arm_position': 'high'})





if __name__ == '__main__':
    main()