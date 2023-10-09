# this file can be delteted. It is just for testing the apriltag detection
import camera
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyapriltags import Detector
from datetime import datetime

if __name__ == "__main__":
  cam = camera.RealSenseCamera(1280,720) 
  camera_params = cam.getCameraParams()
  at_detector = Detector(families='tag16h5',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=1,
                        debug=0)
  
  # detect the tag and get the pose
  tag_size = 5 # centimeters
  while True:
    dt = datetime.now()
    color, depth = cam.read()
    
    tags = at_detector.detect(
      cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), True, camera_params[0:4], tag_size)
    found_tag = False
    for tag in tags:
        if tag.decision_margin < 50: 
           continue
        found_tag = True
        if tag.tag_id != 0:
           continue
      # print the tag pose, tag id, etc well formatted
        # print("Tag Family: ", tag.tag_family)
        print("Tag ID: ", tag.tag_id)
        # print("Tag Hamming Distance: ", tag.hamming)
        # print("Tag Decision Margin: ", tag.decision_margin)
        # print("Tag Homography: ", tag.homography)
        # print("Tag Center: ", tag.center)
        # print("Tag Corners: ", tag.corners)
        print("Tag Pose: ", tag.pose_R, tag.pose_t)
        R = tag.pose_R
        t = tag.pose_t
        # make 4 * 4 transformation matrix
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t.ravel()
        # print("Tag Pose Error: ", tag.pose_err)
        # print("Tag Size: ", tag.tag_size)


    if not found_tag:
      cv2.imshow("color", color)
      if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        exit(0)
      continue
    # #Show the depth image, mask, and filtered depth image
    # cv2.imshow("depth", depth)
    # cv2.imshow("mask", mask.astype(np.uint8)*255)
    # cv2.imshow("filtered depth", filtered_depth)
    # if cv2.waitKey(1) == 27:
    #     cv2.destroyAllWindows()
    #     exit(0)

    # print the time it took to detect the tag well formatted
    print("Time to detect tag: ", datetime.now() - dt)

    for tag in tags:
        if tag.tag_id != 0:
            continue
        if tag.decision_margin < 50:
            continue
        font_scale = 0.5  # Adjust this value for a smaller font size
        font_color = (0, 255, 0)
        font_thickness = 2
        text_offset_y = 30  # Vertical offset between text lines

        # Display tag ID
        cv2.putText(color, str(tag.tag_id), (int(tag.center[0]), int(tag.center[1])), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)

        # Display only the 1 most significant digit for pose_R and pose_t
        pose_R_single_digit = np.round(tag.pose_R[0, 0], 1)  # Round to 1 decimal place
        pose_t_single_digit = np.round(tag.pose_t[0], 1)  # Round to 1 decimal place
   
        cv2.putText(color, str(pose_R_single_digit), (int(tag.center[0]), int(tag.center[1]) + text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)
        cv2.putText(color, str(pose_t_single_digit), (int(tag.center[0]), int(tag.center[1]) + 2 * text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv2.LINE_AA)

    cv2.imshow("color", color)
    if cv2.waitKey(1) == 27:
      cv2.destroyAllWindows()
      exit(0)