import cv2
import numpy as np
import time
import pygame
import sys
from . import lan

class RemoteInterface:
  def __init__(self, name="test-robot"):
    lan.start(name, frame_shape=(360, 1280, 3))
    self.motor_vals = [0] * 10
    self.sensor_vals = np.zeros((20,), np.int32)

    pygame.init()
    self.screen = pygame.display.set_mode((1280, 720))
    self.clock = pygame.time.Clock()
    self.screen.fill((63, 63, 63))

    self.color = None
    self.depth = None

    self.keys = {}
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
      alpha = str.lower(ch)
      self.keys[alpha] = 0

    self.free_frame1 = np.zeros((360, 640, 3), np.uint8)
    self.free_frame2 = np.zeros((360, 640, 3), np.uint8)

  def __del__(self):
    lan.stop()

  def disp1(self, frame):
    self.free_frame1 = frame

  def disp2(self, frame):
    self.free_frame2 = frame

  def _decode_depth_frame(self, frame):
    R = np.left_shift(frame[:, :, 0].astype(np.uint16), 5)
    G = frame[:, :, 1].astype(np.uint16)
    B = np.left_shift(frame[:, :, 2].astype(np.uint16), 5)
    I = np.bitwise_or(R, G, B)
    return I
  
  def depth2rgb(self, depth):
    return cv2.applyColorMap(np.sqrt(depth).astype(np.uint8), cv2.COLORMAP_HSV)
  
  @property
  def motor(self):
    return self.motor_vals
  
  @property
  def sensor(self, idx):
    return self.sensor_vals[idx]
  
  def read(self):
    return self.color, self.depth, np.copy(self.sensor_vals)
  
  def update(self):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit(0)
      elif event.type == pygame.KEYDOWN:
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
          alpha = str.lower(ch)
          keycode = getattr(pygame, f"K_{alpha}")
          if event.key == keycode:
            self.keys[alpha] = 1
      elif event.type == pygame.KEYUP:
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
          alpha = str.lower(ch)
          keycode = getattr(pygame, f"K_{alpha}")
          if event.key == keycode:
            self.keys[alpha] = 0

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
    sensors = lan.sensors()
    if sensors:
      self.sensor_vals = sensors

    pygame.display.flip()

if __name__ == "__main__":
  robot = RemoteInterface()
  while True:
    forward = robot.keys["w"]
    robot.motor[0] = forward * 64
    robot.motor[9] = forward * 64
    robot.update()