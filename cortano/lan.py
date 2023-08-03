import cv2
import numpy as np

import socket
import pickle
import struct
import threading
import json
import time
import requests
import select
import signal
import sys

from multiprocessing import (
  Process,
  Lock,
  Manager,
  Array,
  RawArray,
  RawValue,
  Value
)
import ctypes

_frame_shape = (360, 640)
_color_frame = None
_depth_frame = None
_frame_lock = threading.Lock()
_running = Value(ctypes.c_bool, False)
_connected = RawValue(ctypes.c_bool, False)
_tx_ms_interval = .02 # 100Hz

def _stream_receiver(host, port):
  """
  Handles the connection and processes its stream data.
  """
  global _color_frame, _depth_frame, _frame_lock, _frame_shape, _running, _connected
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  _connected.value = is_connected = False

  payload_size = struct.calcsize('>L')
  data = b""
  gathering_payload = True
  msg_size = 0

  while _running.value:
    if not is_connected:
      data = b""
      gathering_payload = True
      try:
        sock.connect((host, port))
        _connected.value = is_connected = True
      except Exception as e:
        print("Warning:", e)
        time.sleep(1)
        continue

    try:
      ready_to_read, ready_to_write, in_error = select.select(
        [sock,], [sock,], [], 2
      )
      # example ready_to_read: [<socket.socket fd=1332, family=2, type=1, proto=0, laddr=('127.0.0.1', 59746), raddr=('127.0.0.1', 9990)>]
      # example ready_to_write: [<socket.socket fd=1332, family=2, type=1, proto=0, laddr=('127.0.0.1', 59746), raddr=('127.0.0.1', 9990)>]
      if len(ready_to_read) > 0:
        received = sock.recv(4096)
        if received == b'':
          print("Warning: stream received empty bytes, closing and attempting reconnect...")
          sock.close()
          _connected.value = is_connected = False
          continue
        data += received
    except select.error:
      print("Warning: stream has been disconnected for 1.0 seconds, attempting reconnect...")
      sock.shutdown(2)
      sock.close()
      _connected.value = is_connected = False
      continue

    # retry connection...
    if not is_connected: continue

    if gathering_payload:
      if len(data) >= payload_size:
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        gathering_payload = False

    else:
      if len(data) >= msg_size:
        frame_data = data[:msg_size]
        data = data[msg_size:]

        color, depth = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        color = cv2.imdecode(color, cv2.IMREAD_COLOR)
        depth = cv2.imdecode(depth, cv2.IMREAD_UNCHANGED)
        # color, depth = frame[:,:640], frame[:,640:]

        # x1 = np.left_shift(depth[:, :, 1].astype(np.uint16), 8)
        # x2 = depth[:, :, 2].astype(np.uint16)
        # depth = np.bitwise_or(x1, x2)
      
        _frame_lock.acquire()
        _color_frame = color
        _depth_frame = depth
        _frame_lock.release()

        gathering_payload = True

  sock.close()

_host = "0.0.0.0"
_port = 9999

_write_lock = Lock()
_read_lock = Lock()
_motor_values = RawArray(ctypes.c_int, 10)
_sensor_values = RawArray(ctypes.c_int, 20)
_num_sensors = RawValue(ctypes.c_int, 0)
_last_rx_time = RawValue(ctypes.c_double, 0)

_rx_tx_worker = None
_video_worker = None

def _rxtx_worker(host, port, running, wlock, rlock, motor_values, sensor_values, nsensors, rxtime):
  global _tx_ms_interval
  connected = False
  start_time = rxtime.value = last_tx_time = time.time()

  while running.value:
    if not connected:
      try:
        r = requests.get(f"http://{host}:{port}", timeout=1.0)
        connected = True
        last_tx_time = time.time()
      except Exception as e:
        print(e)
        continue

    dt = (time.time() - last_tx_time)
    if dt < _tx_ms_interval:
      time.sleep(_tx_ms_interval - dt)

    try:
      wlock.acquire()
      mvals = motor_values[:]
      wlock.release()
      r = requests.post(f"http://{host}:{port}/update", json={
        "motors": mvals
      }, timeout=0.5)
      last_tx_time = time.time()
      res = r.json()
      if "sensors" in res:
        sensors = res["sensors"]
        rlock.acquire()
        ns = nsensors.value = len(sensors)
        sensor_values[:ns] = sensors
        rxtime.value = last_tx_time - start_time
        rlock.release()

    except Exception as e:
      print(e)
      connected = False

def start(host, port=9999, frame_shape=(360, 640)):
  global _frame_shape, _color_frame, _depth_frame, _host, _port, _running, _video_worker, \
    _write_lock, _read_lock, _motor_values, _sensor_values, _num_sensors, _last_rx_time, _rx_tx_worker
  _host = host
  _port = port
  _frame_shape = frame_shape
  _color_frame = np.zeros((_frame_shape[0], _frame_shape[1], 3), np.uint8)
  _depth_frame = np.zeros((_frame_shape[0], _frame_shape[1]), np.uint16)

  if _running.value:
    print("Warning: stream is already running")
  else:
    _running.value = True
    _video_worker = threading.Thread(target=_stream_receiver, args=(_host, _port))

    _rx_tx_worker = Process(target=_rxtx_worker, args=(
      _host, _port + 1, _running,
      _write_lock, _read_lock, _motor_values, _sensor_values, _num_sensors, _last_rx_time
    ))

    _rx_tx_worker.start()
    _video_worker.start()

def stop():
  global _running, _rx_tx_worker
  _running.acquire()
  running = _running.value
  _running.value = False
  _running.release()
  if running:
    if sys.platform.startswith('win') and _rx_tx_worker:
      # _rx_tx_worker.terminate()
      time.sleep(0.5)
      _rx_tx_worker.kill()
      _rx_tx_worker = None
      time.sleep(0.5)
    else:
      time.sleep(1) # cant kill the process because linux is strange

def sig_handler(signum, frame):
  if signum == signal.SIGINT or signum == signal.SIGTERM:
    stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)

def get_frame():
  global _frame_lock, _color_frame, _depth_frame
  _frame_lock.acquire()
  color = _color_frame
  depth = _depth_frame
  _frame_lock.release()
  return color, depth

def read():
  global _read_lock, _sensor_values, _num_sensors, _last_rx_time
  _read_lock.acquire()
  ns = _num_sensors.value
  if ns > 0:
    sensors = _sensor_values[:ns]
  else:
    sensors = []
  rxt = _last_rx_time.value
  _read_lock.release()
  return [rxt] + sensors # time, values

def write(motor_values):
  global _write_lock, _motor_values
  assert(len(motor_values) == 10)
  motor_values = list(motor_values)
  _write_lock.acquire()
  _motor_values[:] = motor_values
  _write_lock.release()