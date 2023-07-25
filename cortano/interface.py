import cv2
import numpy as np
import time
import pygame
import sys
from . import lan

class RemoteInterface:
  def __init__(self, host="0.0.0.0", port=9999):
    """Remote Interface showing the data coming in from the robot

    Args:
        host (str, optional): host ip of the robot. Defaults to "0.0.0.0".
    """
    lan.start(host, port, frame_shape=(360, 1280, 3))
    self.motor_vals = [0] * 10
    self.sensor_vals = np.zeros((20,), np.int32)

    pygame.init()
    self.screen = pygame.display.set_mode((1280, 720))
    self.clock = pygame.time.Clock()
    self.screen.fill((63, 63, 63))

    self.color = None
    self.depth = None

    self.keys = {k[2:]: 0 for k in dir(pygame) if k.startswith("K_")}
    self.keynames = list(self.keys.keys())

    self.free_frame1 = np.zeros((360, 640, 3), np.uint8)
    self.free_frame2 = np.zeros((360, 640, 3), np.uint8)

    # open3d to viz rgb and depth

  def __del__(self):
    lan.stop()

  def disp1(self, frame):
    """Set an optional output frame to view in disp 1

    Args:
        frame (np.ndarray): frame sized (360, 640, 3) that can be displayed in real time
    """
    self.free_frame1 = frame

  def disp2(self, frame):
    """Set an optional output frame to view in disp 2

    Args:
        frame (np.ndarray): frame sized (360, 640, 3) that can be displayed in real time
    """
    self.free_frame2 = frame

  def _decode_depth_frame(self, frame):
    x1 = np.left_shift(frame[:, :, 1].astype(np.uint16), 8)
    x2 = frame[:, :, 2].astype(np.uint16)
    I = np.bitwise_or(x1, x2)
    return I
  
  def depth2rgb(self, depth):
    """Turn a depth frame into a viewable rgb frame

    Args:
        depth (np.ndarray): depth frame

    Returns:
        np.ndarray: depth frame as color
    """
    return cv2.applyColorMap(np.sqrt(depth).astype(np.uint8), cv2.COLORMAP_JET)
  
  @property
  def motor(self):
    return self.motor_vals
  
  @property
  def sensor(self, idx):
    return self.sensor_vals[idx]
  
  def read(self):
    """Read sensor values from the robot, including color and depth

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): color, depth, other sensor values
    """
    return self.color, self.depth, np.copy(self.sensor_vals)
  
  def update(self):
    """Update the robot by receiving information over WiFi
    """
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        lan.stop()
        sys.exit(0)
      elif event.type == pygame.KEYDOWN:
        for keycode in self.keys.keys():
          if event.key == getattr(pygame, f"K_{keycode}"):
            self.keys[keycode] = 1
      elif event.type == pygame.KEYUP:
        for keycode in self.keys.keys():
          if event.key == getattr(pygame, f"K_{keycode}"):
            self.keys[keycode] = 0

    f = lan.get_frame()
    self.color, depth = f[:,:640], f[:,640:]
    self.depth = self._decode_depth_frame(depth)
    depthrgb = self.depth2rgb(self.depth)
    if f is not None:
      a = (np.swapaxes(np.flip(self.free_frame1, axis=-1), 0, 1))
      b = (np.swapaxes(np.flip(self.free_frame2, axis=-1), 0, 1))
      c = (np.swapaxes(np.flip(self.color, axis=-1), 0, 1))
      d = (np.swapaxes(np.flip(depthrgb, axis=-1), 0, 1))
      a = pygame.surfarray.make_surface(a)
      b = pygame.surfarray.make_surface(b)
      c = pygame.surfarray.make_surface(c)
      d = pygame.surfarray.make_surface(d)
      self.screen.blit(c, (0, 0))
      self.screen.blit(d, (640, 0))
      self.screen.blit(a, (0, 360))
      self.screen.blit(b, (640, 360))

    lan.send({ "motor": self.motor_vals })
    msg = lan.recv()
    if msg and isinstance(msg, dict) and "sensor" in msg:
      self.sensor_vals = msg["sensor"]

    pygame.display.flip()

if __name__ == "__main__":
  robot = RemoteInterface()
  while True:
    forward = robot.keys["w"]
    robot.motor[0] = forward * 64
    robot.motor[9] = forward * 64
    robot.update()