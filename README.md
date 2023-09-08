# Cortano
Remote Interface for CortexNanoBridge

To install on your laptop/desktop:

```bash
# python3 -m venv
sudo apt install python3-venv
python3 -m venv clawbot
source clawbot/bin/activate
python3 -m pip install .
python3 -m pip install . -r /home/sgunnam/sgunnam/clawbot/workshop2/CortexNanoBridge/jetson_nano/requirements.txt # on my desktop
python3 -m pip install . -r /home/nvidia/wsp/CortexNanoBridge/jetson_nano/requirements.txt # on jetson
pip install pyrealsense2
#install cuda
https://developer.nvidia.com/cuda-zone
pip install torch torchvision torchaudio

pip install open3d     
Here is the repository that I am grabbing the tags from as well:
https://github.com/AprilRobotics/apriltag-imgs
```

pip install pytransform3d # visualization

#installation for visualizing the rs camera, decompressing the obj
pip install PyOpenGL
pip install py-lz4framed
pip install lz4

## Example Program
To start a program to connect to your robot, first get the ip address from your Jetson Nano.
This can be done by first going to the Jetson Nano and typing the following into a new terminal:

```bash
ifconfig
```

Your IP Address should be the IPv4 address (ie. 172.168.0.2) under wlan0.

Then, go to your laptop/desktop. After installing this repository (located above), you can now write a program to
connect to your robot:

```python
from cortano import RemoteInterface

if __name__ == "__main__":
  robot = RemoteInterface("172.168.0.2") # remember to put your ip here
  while True:
    robot.update() # required
```

To control a robot, set the motor values (0-10) to anywhere between [-127, 127]

```python
from cortano import RemoteInterface

if __name__ == "__main__":
  robot = RemoteInterface("172.168.0.2")
  while True:
    robot.update()
    
    forward = robot.keys["w"] - robot.keys["s"]
    robot.motor[0] = forward * 64
    robot.motor[9] = forward * 64
```

To get the color and depth frames, as well as other sensor data (up to 20) from the robot,
just read()

```python
from cortano import RemoteInterface

if __name__ == "__main__":
  robot = RemoteInterface("172.168.0.2")
  while True:
    robot.update()
    
    color, depth, sensors = robot.read()
```

That's it!
