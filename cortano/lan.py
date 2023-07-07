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

_frame_shape = (360, 640, 3)
_frame = None
_frame_lock = threading.Lock()

def _try_recv(sock, count=4096):
  try:
    ready_to_read, ready_to_write, in_error = select.select(
      [sock,], [sock,], [], 2
    )
    if len(ready_to_read) > 0:
      received = sock.recv(count)
      if received == b'':
        sock.close()
        print("Warning: stream received empty bytes, closing and attempting reconnect...")
        return -1, ready_to_read, ready_to_write, None
      return len(received), ready_to_read, ready_to_write, received
  except select.error:
    sock.shutdown(2)
    sock.close()
    print("Warning: stream has been disconnected for 1.0 seconds, attempting reconnect...")
    return -1, 0, 0, None
            
def _stream_receiver(host, port,
        buf: RawArray, lock: Lock,
        connected: RawValue, running: RawValue):
  """
  Handles the connection and processes its stream data.
  """
  global _frame, _frame_lock, _frame_shape
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  connected.value = is_connected = False
  payload_size = struct.calcsize('>L')
  data = b""

  while running.value: # instead of killing the thread, refresh the socket
    if not is_connected:
      data = b""
      try:
        sock.connect((host, port))
        connected.value = is_connected = True
      except Exception as e:
        print("Warning:", e)
        time.sleep(1)
        continue

    while len(data) < payload_size:
      length, _, __, received = _try_recv(sock)
      if length == -1:
        connected.value = is_connected = False # attempt to reconnect
        break
      elif length > 0:
        data += received
    # retry connection...
    if not is_connected: continue

    # we have successfully read a message, do something with it
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]

    msg_size = struct.unpack(">L", packed_msg_size)[0]

    while len(data) < msg_size:
      length, _, __, received = _try_recv(sock)
      if length == -1:
        connected.value = is_connected = False # attempt to reconnect
      elif length > 0:
        data += received
    # retry connection...
    if not is_connected: continue

    frame_data = data[:msg_size]
    data = data[msg_size:]

    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    _frame_lock.acquire()
    _frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)#.reshape((-1,))
    _frame_lock.release()

  sock.close()

def _tx_worker(host, port,
        txbuf: RawArray, txlen: RawValue, txlock: Lock, txinterval,
        connected: RawValue, running: RawValue):
  tx_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  is_connected = False
  last_tx_timestamp = time.time()

  while running.value:
    if not is_connected:
      try:
        tx_socket.connect((host, port))
        is_connected = True
      except Exception as e:
        print("Warning:", e)
        time.sleep(1)
        continue

    curr_time = time.time()
    if curr_time - last_tx_timestamp >= txinterval:
      last_tx_timestamp = curr_time

      txlock.acquire()
      if txlen.value == 0:
        txlock.release() # do nothing
        continue
      tx = bytearray(txbuf[:txlen.value]).decode()
      txlock.release()

      tx = pickle.dumps(tx, 0)
      size = len(tx)

      try:
          tx_socket.sendall(struct.pack('>L', size) + tx)
      except ConnectionResetError:
          is_connected = False
          tx_socket.close()
          # running.value = False
      except ConnectionAbortedError:
          is_connected = False
          tx_socket.close()
          # running.value = False
      except BrokenPipeError:
          is_connected = False
          tx_socket.close()
          # running.value = False

    time.sleep(0.001) # throttle to prevent overusage

  tx_socket.close()

def _rx_worker(host, port,
        rxbuf: RawArray, rxlen: RawValue, rxlock: Lock, rxtime: RawValue,
        connected: RawValue, running: RawValue):
  rx_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  payload_size = struct.calcsize('>L')
  is_connected = False
  data = b""

  while running.value:
    if not is_connected:
      data = b""
      try:
        rx_socket.connect((host, port))
        is_connected = True
      except Exception as e:
        print("Warning:", e)
        time.sleep(1)
        continue

    while len(data) < payload_size:
      length, _, __, received = _try_recv(rx_socket)
      if length == -1:
        is_connected = False # attempt to reconnect
      elif length > 0:
        data += received
    # retry connection...
    if not is_connected: continue

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]

    msg_size = struct.unpack(">L", packed_msg_size)[0]

    while len(data) < msg_size:
      length, _, __, received = _try_recv(rx_socket)
      if length == -1:
        is_connected = False # attempt to reconnect
      elif length > 0:
        data += received
    # retry connection...
    if not is_connected: continue

    rx = data[:msg_size]
    data = data[msg_size:]

    rx = pickle.loads(rx, fix_imports=True, encoding="bytes")
    rxlock.acquire()
    bytearr = rx.encode()
    rxlen.value = len(bytearr)
    rxbuf[:len(rx)] = bytearr
    rxtime.value = time.time()
    rxlock.release()

  rx_socket.close()

_host = "0.0.0.0"
_port = 9999
_encoding_parameters = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
_tx_ms_interval = .02 # 50Hz

_running = RawValue(ctypes.c_bool, False)
_connected = RawValue(ctypes.c_bool, False)

_tx_buf = RawArray(ctypes.c_uint8, 128)
_tx_len = RawValue(ctypes.c_int32, 0)
_tx_lock = Lock()
_rx_buf = RawArray(ctypes.c_uint8, 128)
_rx_len = RawValue(ctypes.c_int32, 0)
_rx_lock = Lock()

_rx_timestamp = RawValue(ctypes.c_float, 0.0)
_processes = []
_stream_thread = None

def start(host, port=9999, frame_shape=(360, 640, 3)):
  global _frame_shape, _frame, _host, _port, _running, _stream_thread
  _host = host
  _port = port
  _frame_shape = frame_shape
  _frame = np.zeros((_frame_shape), np.uint8)

  if _running.value:
    print("stream is already running")
  else:
    _running.value = True
    _stream_thread = threading.Thread(target=_stream_receiver, args=(
      _host, _port,
      _frame, _frame_lock,
      _connected, _running
    ))

    _processes.append(Process(target=_tx_worker, args=(
      _host, _port + 2,
      _tx_buf, _tx_len, _tx_lock, _tx_ms_interval,
      _connected, _running
    )))
    _processes.append(Process(target=_rx_worker, args=(
      _host, _port + 1,
      _rx_buf, _rx_len, _rx_lock, _rx_timestamp,
      _connected, _running
    )))

    _stream_thread.start()
    for p in _processes:
      p.start()

def stop():
  global _running, _processes
  if _running.value:
    _running.value = False
    time.sleep(1)
    for p in _processes:
      p.kill()
    time.sleep(1)
    for p in _processes:
      p.kill()
    time.sleep(1)

def set_frame(frame: np.ndarray):
  global _frame_lock, _frame
  _frame_lock.acquire()
  _frame = frame
  _frame_lock.release()
  
def get_frame():
  global _frame_lock, _frame
  _frame_lock.acquire()
  frame = _frame
  _frame_lock.release()
  return frame

def recv():
  global _rx_lock, _rx_buf, _rx_len
  _rx_lock.acquire()
  if _rx_len.value == 0:
    _rx_lock.release()
    return None
  rx = bytearray(_rx_buf[:_rx_len.value])
  _rx_lock.release()
  msg = json.loads(rx.decode())
  return msg

def sensors():
  msg = recv()
  if msg and isinstance(msg, dict) and "sensors" in msg:
    return msg["sensors"]
  return None

def send(msg):
  global _tx_lock, _tx_buf, _tx_len
  _tx_lock.acquire()
  bytearr = json.dumps(msg).encode()
  _tx_buf[:len(bytearr)] = bytearr
  _tx_len.value = len(bytearr)
  _tx_lock.release()