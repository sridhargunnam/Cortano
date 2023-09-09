(clawbot) nvidia@ubuntu:/media/nvidia/data/wsp/clawbot/Cortano/example$ python realsense_rgbd_odometry.py 
Traceback (most recent call last):
  File "realsense_rgbd_odometry.py", line 57, in <module>
    depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)
RuntimeError: set_xu(...). xioctl(UVCIOC_CTRL_QUERY) failed Last Error: Protocol error


(clawbot) nvidia@ubuntu:/media/nvidia/data/wsp/clawbot/Cortano/example$ python realsense_rgbd_odometry.py 
Traceback (most recent call last):
  File "realsense_rgbd_odometry.py", line 57, in <module>
    depth_sensor.set_option(rs.option.visual_preset, Preset.HighAccuracy)
RuntimeError: get_xu(ctrl=1) failed! Last Error: Device or resource busy