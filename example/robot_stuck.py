import cv2
import numpy as np

def is_robot_stuck(image1, image2, feature_threshold=10, movement_threshold=5):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors with ORB
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance (best first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Check if enough matches are found
    if len(matches) > feature_threshold:
        # Extract location of good matches
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        # Compute homography
        H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Check if homography is close to identity matrix
        identity = np.identity(3)
        if np.linalg.norm(H - identity) < movement_threshold:
            return True  # Robot is likely stuck
        else:
            return False  # Robot is likely moving
    else:
        return False  # Not enough matches to make a decision

import camera 
import time
def main():
    rsCamera = camera.RealSenseCamera(1280, 720, 30)
    # capture images 10 seconds apart and check if robot is stuck
    #check the fps 
    debug = False
    start = time.time()
    while True:
        while True:
            image1, _ = rsCamera.read()
            if image1 is not None:
                break
        # time.sleep(10)
        while True:
            image2, _ = rsCamera.read()
            if image2 is not None:
                break
        stuck = is_robot_stuck(image1, image2)
        if stuck:
            print("Robot is stuck")
        else:
            print("Robot is moving")
        # stack image and show

        end_time = time.time()
        print("FPS: ", 1/(end_time - start))
        start = end_time
        #Scalling the image to 25% to fit the screen
        if debug == True:
            image = np.hstack((image1, image2))
            image = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
            cv2.imshow('image', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                exit(0)

        
if __name__ == '__main__':
    main()