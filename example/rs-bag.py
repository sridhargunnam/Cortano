import pyrealsense2 as rs
import numpy as np
import cv2

# Replace 'your_file.bag' with the path to your saved .bag file
bag_filename = '/home/sgunnam/Documents/20230906_080949.bag'


pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_filename)
pipeline.start(config)

while True:
    try:
        # Wait for a new frame
        frames = pipeline.wait_for_frames()
        if not frames:
            break  # No more frames, exit the loop

        # get depth and color frames
        for frame in frames:
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            dWidth = depth_frame.get_width()
            dHeight = depth_frame.get_height()
            cHeight = color_frame.get_height()
            cWidth = color_frame.get_width()


            #view the depth image and color image using openCV in a same window
            #create a numpy array with the correct dimensions from the raw data
            depth_array = np.asanyarray(depth_frame.get_data())
            color_array = np.asanyarray(color_frame.get_data())
            #use numpy to create an image from the depth array
            # visualize depth image as a color image using colormap
            depth_cv = cv2.applyColorMap(cv2.convertScaleAbs(depth_array, alpha=0.03), cv2.COLORMAP_JET)



            #use numpy to create an image from the color array
            color_cv = cv2.cvtColor(color_array, cv2.COLOR_BGR2RGB)
            #stack depth and color images horizontally
            # make color image of same size as depth image
            color_cv = cv2.resize(color_cv, (dWidth, dHeight))          
            images = np.hstack((color_cv, depth_cv))
            #show the stacked images
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)



    except KeyboardInterrupt:
        break  # Exit the loop on Ctrl+C

# Stop the pipeline
pipeline.stop()
