# installing ssd on jetson agx xavier
on the jetson device
https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/FlashingSupport.html#to-set-up-an-nvme-drive-manually-for-booting

lsblk -d -p | grep nvme | cut -d\  -f 1

sudo parted /dev/nvme0n1 mklabel gpt

sudo parted /dev/nvme0n1 mkpart APP 0GB 999GB

on host:
https://github.com/jetsonhacks/bootFromExternalStorage

./get_jetson_files.sh

https://forums.developer.nvidia.com/t/controlling-app-size-partition-jetpack-5-1-2/265484

cd bootFromExternalStorage/R35.4.1/Linux_for_Tegra

sudo ./tools/kernel_flash/l4t_initrd_flash.sh --external-device nvme0n1p1 -c ./tools/kernel_flash/flash_l4t_external.xml  --showlogs --network usb0 jetson-agx-xavier-devkit internal

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

pip install open3d     # this won't install with cuda support on arm64, I will need to manually comiple 
Here is the repository that I am grabbing the tags from as well:
https://github.com/AprilRobotics/apriltag-imgs
pip install pyapriltags
source /home/nvidia/wsp/clawbot/sudo 
pip3 install -U jetson-stats
clawbot/bin/activate
```

pip install pytransform3d # visualization

#cupoch
sudo apt-get install libxinerama-dev libxcursor-dev libglu1-mesa-dev
pip3 install cupoch
pip install matplotlib



#installation for visualizing the rs camera, decompressing the obj
pip install PyOpenGL
pip install py-lz4framed
pip install lz4

# jetson inference 
to make jetson inference work in the virtual env I needed to manually copy the site-packages to virtual env
cp  /usr/lib/python3.8/dist-packages/jetson*  /home/nvidia/wsp/clawbot/clawbot/lib/python3.8/site-packages/ -r 
https://github.com/dusty-nv/jetson-inference/issues/1285#issuecomment-1197359239

docker:
sudo docker pull nvcr.io/nvidia/l4t-ml:r35.2.1-py3
sudo docker run -it --rm --runtime nvidia --network host -v /home/nvidia/wsp/clawbot:/home/nvidia/wsp/clawbot nvcr.io/nvidia/l4t-ml:r35.2.1-py3
xhost +local:root
sudo docker run -it --privileged --rm --device-cgroup-rule "c 81:* rmw"  --device-cgroup-rule "c 189:* rmw" --runtime nvidia --network host -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all  -v /dev:/dev  -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v /tmp/.docker.xauth:/tmp/.docker.xauth:rw  -v /home/nvidia/wsp/clawbot:/home/nvidia/wsp/clawbot nvcr.io/nvidia/l4t-ml:r35.2.1-py3

sudo apt update
sudo apt install vim
pip3 install open3d pyrealsense2 pygame pyapriltags

sudo docker run --privileged --name realsense-2-container --rm -p 8084:8084 -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev:ro --gpus all -it lmwafer/realsense-ready:2.0-ubuntu18.04

bash
cd /home/nvidia/wsp/clawbot/Cortano
. docker/bin/activate 
## TODO
1. Integrate IMU + RGBD odometry
2. Integrate april tags based pose correction
3. PID testing

# for GPU and CPU stats
 tegrastats --interval 1 --logfile tegrastats.txt 
slam.py is ~1 fps, gpu slam is ~3 fps

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
