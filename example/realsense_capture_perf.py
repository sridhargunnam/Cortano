csv_dat = "/media/nvidia/data/wsp/librealsense/build/tools/data-collect/log.csv"
import csv
from collections import defaultdict


with open(csv_dat) as f:
    #skip empty lines, lines starting with #
    lines = [line for line in f if line.strip() and not line.startswith('#')]
    # create a dictionary of lists, key is the first field in the line, and the key can be repeated. 
    # the values are the rest of the fields in the line
    d = defaultdict(list)
    for line in lines:
        key, *rest = line.split(',')
        d[key].append(rest)
    # print the dictionary keys
    print(d.keys()) # this prints as dict_keys(['Depth', 'Color', 'Gyro', 'Accel'])
    # the remaing fields are streamType,frameIndex,frameNumber,HWTimestamp_ms,HostTimestamp_ms, and other fields
    # find the closest match to the HWTimestamp_ms from the Depth stream with that for other streams 
    # and print the difference in timestamps
    for i in range(len(d['Depth'])):
        # get the HWTimestamp_ms from the Depth stream
        depth_ts = float(d['Depth'][i][3])
        # get the closest match to the HWTimestamp_ms from the other streams
        # the other streams have different number of frames, so we need to find the closest match
        # to the depth_ts
        # find the closest match to depth_ts in the Color stream
        color_ts = min(d['Color'], key=lambda x: abs(float(x[3]) - depth_ts))
        # find the closest match to depth_ts in the Gyro stream
        gyro_ts = min(d['Gyro'], key=lambda x: abs(float(x[3]) - depth_ts))
        # find the closest match to depth_ts in the Accel stream
        accel_ts = min(d['Accel'], key=lambda x: abs(float(x[3]) - depth_ts))
        # print the difference in timestamps in a single line

        print("depth_ts: ", depth_ts)
        print("color_ts: ", color_ts[3])
        print("gyro_ts: ", gyro_ts[3])
        print("accel_ts: ", accel_ts[3])
        print("color_ts - depth_ts: ", float(color_ts[3]) - depth_ts)
        print("gyro_ts - depth_ts: ", float(gyro_ts[3]) - depth_ts)
        print("accel_ts - depth_ts: ", float(accel_ts[3]) - depth_ts)
        # print("")

