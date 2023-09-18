import pygame
import sys
import camera
import time
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def RotZ(angle):
  return R.from_euler("z", [angle], degrees=True)

def raster(pts, scale, off=400):
  if len(pts.shape) == 1:
    p = pts[:2] * scale
    p[0] += off
    p[1] = off - p[1]
  else:
    p = pts[:,:2] * scale
    p[:,0] += off
    p[:,1] = off - p[:,1]
  return p

class SimInterface:
  def __init__(self):
    self.motor_vals = np.zeros((10,), np.int32)
    self.sensor_vals = np.zeros((3,), np.float32)

    pygame.init()
    self.screen = pygame.display.set_mode((1600, 800))
    self.clock = pygame.time.Clock()
    self.screen.fill((63, 63, 63))

    self.color = None
    self.depth = None

    self.keys = {k[2:]: 0 for k in dir(pygame) if k.startswith("K_")}

    # cam = camera.RealSenseCamera()

    self.angle = 0.0
    self.position = np.array([0, 36, 8], np.float64)
    self.speed_coeff = 0.15
    self.angle_coeff = .2

    self.pitch = -33
    self.grip = 0

    self.pitch_coeff = 0.5
    self.pitch_v = 0.0
    self.pitch_acc = -0.003
    self.pitch_range = [-33, 30]
    self.pitch_v_lim = -.3
    self.start_time = time.time()
    self.weighted_coeff = 0.08

    self.has_ball = False
    self.can_grab = False
    self.ball_location = None
    self.claw_pos = np.array([0, 0, 0], np.float64)
    self.claw_angle = 0
    self.claw_range = [0, 75]
    self.closed_lim = 15
    self.open_lim = 30
    self.claw_coeff = 1.5

    self.goal = None

  @property
  def motor(self): return self.motor_vals
  @property
  def sensor(self): return self.sensor_vals
  @property
  def pos(self): return self.position[:2]

  def update_motors(self):
    self.motor_vals = np.clip(self.motor_vals, -127, 127)

    left = -self.motor_vals[0]
    right = self.motor_vals[9]

    turn = (right - left) / 127
    turn_dir = turn < 0
    turn = np.abs(turn) - .5
    if turn < 0:
      turn = 0
    if turn_dir:
      turn = -turn
    forward = (left + right) / 2 / 127

    vel = forward - np.abs(turn) / 2
    angle = turn * self.angle_coeff

    self.angle += angle / 2
    self.position += RotZ(self.angle).apply(
      np.array([vel * self.speed_coeff, 0, 0], np.float64))[0]
    self.angle += angle / 2

    if self.position[0] < -72 + 5:
      self.position[0] = -72 + 5
    elif self.position[0] > 72 - 5:
      self.position[0] = 72 - 5

    if self.position[1] < 0 + 5:
      self.position[1] = 0 + 5
    elif self.position[1] > 72 - 5:
      self.position[1] = 72 - 5
    
    arm = self.motor_vals[1]
    pitch_v = arm / 127 * self.pitch_coeff
    if pitch_v == 0.0:
      self.pitch_v += self.pitch_acc
      if self.pitch_v < self.pitch_v_lim:
        self.pitch_v = self.pitch_v_lim
      self.pitch = self.pitch + self.pitch_v
    else:
      pitch_v -= self.weighted_coeff
      if pitch_v > 0.0 and pitch_v < 0.05:
        pitch_v = 0.0
      self.pitch = self.pitch + pitch_v
      self.pitch_v = 0.0

    if self.pitch < self.pitch_range[0]:
      self.pitch = self.pitch_range[0]
      self.pitch_v = 0.0
    elif self.pitch > self.pitch_range[1]:
      self.pitch = self.pitch_range[1]

    claw = self.motor_vals[2]
    claw_v = claw / 127 * self.claw_coeff
    self.claw_angle = self.claw_angle + claw_v
    if self.claw_angle < self.claw_range[0]:
      self.claw_angle = self.claw_range[0]
    elif self.claw_angle > self.claw_range[1]:
      self.claw_angle = self.claw_range[1]

    self.claw_pos = R.from_euler("x", [-self.pitch], degrees=True).apply(
      np.array([0, 12, 0], np.float64))
    self.claw_pos = self.claw_pos[0] + np.array([0, -4.5, -7.5], np.float64)
    self.claw_pos += np.array([0, 3.5, 0], np.float64)

    is_lowered = np.abs(self.pitch - self.pitch_range[0]) < 4
    is_now_closed = self.claw_angle <= self.closed_lim

    if self.ball_location is not None:
      ball_within = np.linalg.norm(self.ball_location[:2] - self.claw_pos) < 0.8
      if self.can_grab and is_now_closed and is_lowered:
        self.can_grab = False
        if ball_within:
          self.has_ball = True

    self.can_grab = (self.can_grab and not is_now_closed) \
      or self.claw_angle >= self.open_lim # in the future
    if not is_now_closed: self.has_ball = False
    if self.has_ball and self.claw_angle <= self.closed_lim:
      self.claw_angle = self.closed_lim

  def draw_map(self, surface):
    wall_color = (128, 128, 128)
    robot_color = (0, 0, 255)
    wheel_color = (0, 127, 0)
    claw_color = (255, 0, 0)
    # goal_color = (255, 255, 0)  # Yellow
    scale = 4

    pygame.draw.rect(surface, wall_color,
      pygame.Rect(111, 111, 578, 578), width=3)
    pygame.draw.line(surface, wall_color, (110, 400), (688, 400), 3)

    corners = np.array([
      [-5, -5, 0],
      [-5,  5, 0],
      [ 5,  5, 0],
      [ 5, -5, 0]], np.float64)
    wheel = np.array([
      [-2, -.5, 0],
      [-2,  .5, 0],
      [ 2,  .5, 0],
      [ 2, -.5, 0]], np.float64)
    offsets = np.array([
      [-4, -5.5, 0],
      [-4,  5.5, 0],
      [ 4,  5.5, 0],
      [ 4, -5.5, 0]], np.float64)
    
    corners = RotZ(self.angle).apply(corners) + self.position
    pygame.draw.polygon(surface, robot_color, raster(corners, scale), 2)
    for offset in offsets:
      this_wheel = wheel + offset
      this_wheel = RotZ(self.angle).apply(this_wheel) + self.position
      pygame.draw.polygon(surface, wheel_color, raster(this_wheel, scale))

    claw_endpt = RotZ(self.angle - 90).apply(self.claw_pos)[0]
    claw_endpt = claw_endpt + self.position
    pygame.draw.circle(surface, claw_color, raster(claw_endpt, scale), 10, 2)

  def draw_robot(self, surface):
    robot_color = (0, 0, 255)
    claw_color = (255, 0, 0)
    wheel_color = (0, 127, 0)
    scale = 20
    down = 1

    wheel1 = np.array((-4, .25 + down), np.float64)
    wheel2 = np.array(( 4, .25 + down), np.float64)

    pygame.draw.rect(surface, robot_color,
      pygame.Rect(-5 * scale + 400, (-.5 + down) * scale + 400, scale * 10, scale), 1)
    pygame.draw.rect(surface, robot_color,
      pygame.Rect(-5 * scale + 400, (-8 + down) * scale + 400, scale, scale * 7.5), 1)
    pygame.draw.circle(surface, wheel_color, wheel1 * scale + 400, 2 * scale, 1)
    pygame.draw.circle(surface, wheel_color, wheel2 * scale + 400, 2 * scale, 1)

    arm_link = np.array([
      [-6.25, -.25, 0],
      [ 6.25, -.25, 0],
      [ 6.25,  .25, 0],
      [-6.25,  .25, 0]], np.float64)
    arm_link = RotZ(-self.pitch).apply(arm_link)
    arm_link += RotZ(-self.pitch).apply(np.array([6, 0, 0], np.float64).T)

    arm_link1 = arm_link + np.array([-4.5, -7.5 + down, 0], np.float64)
    arm_link2 = arm_link + np.array([-4.5, -5.5 + down, 0], np.float64)
    
    pygame.draw.polygon(surface, robot_color, arm_link1[:,:2] * scale + 400, 1)
    pygame.draw.polygon(surface, robot_color, arm_link2[:,:2] * scale + 400, 1)

    claw_td = np.array([
      [ -.5, 0, 0],
      [-1.5, -1, 0],
      [-1.5, -2, 0],
      [ -.5, -3, 0],
      [   0, -3, 0],
      [-1.0, -2, 0],
      [-1.0, -1, 0],
      [   0, 0, 0]
    ], np.float64)
    claw_td = RotZ(-self.claw_angle).apply(claw_td)
    claw_td[:,1] += 12

    claw_td1 = claw_td
    claw_td2 = claw_td.copy()
    claw_td2[:,0] = -claw_td2[:,0]
    pygame.draw.polygon(surface, claw_color, claw_td1[:,:2] * scale + 400, 1)
    pygame.draw.polygon(surface, claw_color, claw_td2[:,:2] * scale + 400, 1)
    pygame.draw.rect(surface, claw_color,
                     pygame.Rect(-1 * scale + 400, 12 * scale + 400, 2 * scale, 2 * scale), 1)
    
    c_f = -1.5 + np.cos(np.radians(self.claw_angle)) * 3
    claw_link = np.array([
      [-3.5, -.5, 0],
      [-3.5, 2.5, 0],
      [-2.5, 2, 0],
      [ c_f, 2, 0],
      [ c_f, 0, 0],
      [-2.5, 0, 0]], np.float64)
    claw_link += np.array([self.claw_pos[1], self.claw_pos[2] + down, 0], np.float64)
    pygame.draw.polygon(surface, claw_color, claw_link[:,:2] * scale + 400, 1)

  def update_sensors(self):
    # self.sensor_vals[0] = time.time() - self.start_time
    self.sensor_vals[0] = 700 + int((self.pitch + 30) / 60 * 1925)
    self.sensor_vals[1] = self.has_ball + 0
    self.sensor_vals[2] = self.has_ball + 0

  def set_goal(self, goal_position, goal_angle):
      self.goal = {"position": goal_position, "angle": goal_angle}

  def draw_goal(self, surface):
      if self.goal:
          goal_color = (255, 0, 0)  # red
          scale = 4
          goal_position = self.goal["position"]
          goal_angle = self.goal["angle"]

          goal_position = raster(goal_position, scale)
          
          pygame.draw.circle(surface, goal_color, goal_position, 10)
          goal_direction = np.array([np.cos(np.radians(goal_angle)), np.sin(np.radians(goal_angle))])
          goal_endpoint = goal_position + (goal_direction * 20)
          pygame.draw.line(surface, goal_color, goal_position, goal_endpoint, 2)

  def update(self):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        sys.exit(0)
      elif event.type == pygame.KEYDOWN:
        for keycode in self.keys.keys():
          if event.key == getattr(pygame, f"K_{keycode}"):
            self.keys[keycode] = 1
      elif event.type == pygame.KEYUP:
        for keycode in self.keys.keys():
          if event.key == getattr(pygame, f"K_{keycode}"):
            self.keys[keycode] = 0

    pygame.draw.rect(self.screen, (255, 255, 255),
                     pygame.Rect(0, 0, 800, 800))
    
    map_surface  = np.ones((800, 800, 3), np.uint8) * 255
    side_surface = np.ones((800, 800, 3), np.uint8) * 255
    
    a = pygame.surfarray.make_surface(map_surface)
    b = pygame.surfarray.make_surface(side_surface)

    self.update_motors()
    self.update_sensors()
    self.draw_map(a)
    self.draw_robot(b)
    self.draw_goal(b)  # Add this line to draw the goal

    # Annotate the goal, and robot position and angle
    font = pygame.font.SysFont("Arial", 20)
    #lambda function to convert the position and angle to a string with 2 decimal places
    position_to_string = lambda x: f"{x[0]:.2f}, {x[1]:.2f}"
    angle_to_string = lambda x: f"{x:.2f}"
    #position and angle of the robot
    position = position_to_string(self.position)
    angle = angle_to_string(self.angle)
    #position and angle of the goal
    goal_position = position_to_string(self.goal["position"])
    goal_angle = angle_to_string(self.goal["angle"])
    #text to display
    # text = f"Robot Position: {position}\nRobot Angle: {angle}\nGoal Position: {goal_position}\nGoal Angle: {goal_angle}"
    #render the text
    # text_surface = font.render(text, True, (0, 0, 0))
    #blit the text to the side surface
    # b.blit(text_surface, (0, 0))


    
    goal_position_text  = font.render(f"Goal Position  : {position}", True, (0, 0, 0))
    robot_position_text = font.render(f"Robot Position : {goal_position}", True, (0, 0, 0))
    goal_angle_text     = font.render(f"Goal Angle     : {angle}", True, (0, 0, 0))
    robot_angle_text    = font.render(f"Robot Angle    : {goal_angle}", True, (0, 0, 0))

    # Add the text to the side surface
    b.blit(goal_position_text, (0, 0))
    b.blit(robot_position_text, (0, 20))
    b.blit(goal_angle_text, (0, 40))
    b.blit(robot_angle_text, (0, 60))
  


    self.screen.blit(a, (0, 0))
    self.screen.blit(b, (800, 0))

    pygame.display.flip()

  def read(self):
    return self.sensor_vals