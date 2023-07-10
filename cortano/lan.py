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

_source = False
_frame_shape = (360, 640, 3)
_frame = None
_frame_lock = threading.Lock()
_running = RawValue(ctypes.c_bool, False)
_connected = RawValue(ctypes.c_bool, False)
_encoding_parameters = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
_tx_ms_interval = .02 # 50Hz

def _stream_receiver(host, port):
  """
  Handles the connection and processes its stream data.
  """
  global _frame, _frame_lock, _frame_shape, _running, _connected
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

        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        _frame_lock.acquire()
        _frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        _frame_lock.release()

        gathering_payload = True

  sock.close()

def _rxtx_worker(host, port, running: RawValue,
        rxbuf: RawArray, rxlen: RawValue, rxlock: Lock, rxtime: RawValue,
        txbuf: RawArray, txlen: RawValue, txlock: Lock, source):
  is_connected = False

  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

  payload_size = struct.calcsize('>L')
  data = b""
  gathering_payload = True # states: gathering_payload, gathering_msg
  msg_size = 0
  txtime = time.time()

  while running.value:
    if not is_connected:
      data = b""
      gathering_payload = True
      try:
        sock.connect((host, port))
        conn = sock
        is_connected = True
      except Exception as e:
        print("Warning:", e)
        time.sleep(1)
        continue

    curr_time = time.time()

    try:
      ready_to_read, ready_to_write, in_error = select.select(
        [conn,], [conn,], [], 2
      )
      if len(ready_to_read) > 0:
        received = conn.recv(4096)
        if received == b'':
          print("Warning: RxTx received empty bytes, closing and attempting reconnect...")
          conn.close()
          is_connected = False
          continue
        data += received
    except select.error:
      print("Warning: RxTx has been disconnected for 1.0 seconds, attempting reconnect...")
      conn.shutdown(2)
      conn.close()
      is_connected = False
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
        rx = data[:msg_size]
        data = data[msg_size:]

        rx = pickle.loads(rx, fix_imports=True, encoding="bytes")
        rxlock.acquire()
        bytearr = rx.encode()
        rxlen.value = len(bytearr)
        rxbuf[:len(rx)] = bytearr
        rxtime.value = curr_time
        rxlock.release()

        gathering_payload = True

    if (curr_time - txtime) >= _tx_ms_interval:
      txtime = curr_time

      txlock.acquire()
      if txlen.value == 0:
        txlock.release() # do nothing
        continue
      tx = bytearray(txbuf[:txlen.value]).decode()
      txlock.release()

      tx = pickle.dumps(tx, 0)
      size = len(tx)

      try:
          conn.sendall(struct.pack('>L', size) + tx)
      except ConnectionResetError:
          is_connected = False
          conn.close()
      except ConnectionAbortedError:
          is_connected = False
          conn.close()
      except BrokenPipeError:
          is_connected = False
          conn.close()

  sock.close()

_host = "0.0.0.0"
_port = 9999

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
  if _source:
    _host = "0.0.0.0"
  else:
    _host = host
  
  _port = port
  _frame_shape = frame_shape
  _frame = np.zeros((_frame_shape), np.uint8)

  if _running.value:
    print("Warning: stream is already running")
  else:
    _running.value = True
    _stream_thread = threading.Thread(target=_stream_receiver, args=(_host, _port))

    _processes.append(Process(target=_rxtx_worker, args=(
      _host, _port + 1, _running,
      _rx_buf, _rx_len, _rx_lock, _rx_timestamp,
      _tx_buf, _tx_len, _tx_lock, _source
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

def sig_handler(signum, frame):
  if signum == signal.SIGINT or signum == signal.SIGTERM:
    stop()
  sys.exit()

signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)

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

def send(msg):
  global _tx_lock, _tx_buf, _tx_len
  _tx_lock.acquire()
  bytearr = json.dumps(msg).encode()
  _tx_buf[:len(bytearr)] = bytearr
  _tx_len.value = len(bytearr)
  _tx_lock.release()