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

# def send_command(command, args):
#     try:
#       print("sending ", command, args)
#       client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#       client_socket.connect(('localhost', 6000))
#       command_data = json.dumps({'command': command, 'args': args})
#       client_socket.send(command_data.encode())
#       client_socket.close()
#     except:
#       # print("failed to send", command, args)
#       pass

def send_command(command, args=None):
    response = None 
    try:
      print("sending ", command, args)
      client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      client_socket.connect(('localhost', 6000))
      command_data = json.dumps({'command': command, 'args': args})
      client_socket.send(command_data.encode())
      # Waiting for response
      response_data = client_socket.recv(1024)
      response = json.loads(response_data.decode())
      client_socket.close()
    except Exception as e:
      print(f"Failed to send command or receive response: {e}")
      pass
    return response
import time

def catch_object():
    send_command('drive', ['forward', 30, 0.8])
    # # time.sleep(2)
    send_command('claw', [20, 'close', 1, 1])
    # # time.sleep(2)
    send_command('drive', ['backward', 30, 0.2])
    send_command('claw', [20, 'close', 1, 1])
    # time.sleep(2)
      # armPosition=ARM_POSITION.low, motor=2, error=20
    send_command('update_robot_move_arm', ['high', 2, 20])
    # send_command('update_robot_move_arm', ['low', 2, 20])
    send_command("stop_drive", None)

   
def main():
    send_command('drive', ['forward', 30, 0.3])
    # # time.sleep(2)
    send_command('claw', [20, 'close', 1, 1])
    # # time.sleep(2)
    send_command('drive', ['backward', 30, 0.2])
    send_command('claw', [20, 'close', 1, 1])
    # time.sleep(2)
      # armPosition=ARM_POSITION.low, motor=2, error=20
    send_command('update_robot_move_arm', ['high', 2, 20])
    # send_command('update_robot_move_arm', ['low', 2, 20])
    send_command("stop_drive", None)
    send_command('update_robot_move_arm', ['low', 2, 20])
    send_command('claw', [20, 'open', 1, 3])




if __name__ == '__main__':
    main()