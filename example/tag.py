import camera
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyapriltags import Detector
from datetime import datetime

if __name__ == "__main__":
  cam = camera.RealSenseCamera()
  camera_params = cam.getCameraParams()
  at_detector = Detector(families='tag16h5',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)
  
  # detect the tag and get the pose
  while True:
    dt = datetime.now()
    color, depth = cam.read()
    
    tags = at_detector.detect(
      cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), True, camera_params[0:4], 2.5)
    found_tag = False
    for tag in tags:
        if tag.decision_margin < 50: 
           continue
        found_tag = True
      # print the tag pose, tag id, etc well formatted
        print("Tag Family: ", tag.tag_family)
        print("Tag ID: ", tag.tag_id)
        print("Tag Hamming Distance: ", tag.hamming)
        print("Tag Decision Margin: ", tag.decision_margin)
        print("Tag Homography: ", tag.homography)
        print("Tag Center: ", tag.center)
        print("Tag Corners: ", tag.corners)
        print("Tag Pose: ", tag.pose_R, tag.pose_t)
        print("Tag Pose Error: ", tag.pose_err)
        print("Tag Size: ", tag.tag_size)


    if not found_tag:
       continue
    # print the time it took to detect the tag well formatted
    print("Time to detect tag: ", datetime.now() - dt)

    for tag in tags:
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
        exit(0)


    # for tag in tags:
    #     if tag.decision_margin < 50: 
    #         continue
    #     cv2.putText(color, str(tag.tag_id), (int(tag.center[0]), int(tag.center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #     cv2.polylines(color, [np.int32(tag.corners)], True, (0, 255, 0), 2)
    #     cv2.putText(color, str(tag.pose_R), (int(tag.center[0]), int(tag.center[1]) + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #     cv2.putText(color, str(tag.pose_t), (int(tag.center[0]), int(tag.center[1]) + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.imshow("color", color)
    # if cv2.waitKey(1) == 27:
    #     exit(0)