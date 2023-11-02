import cv2
import numpy as np
import time 
import camera

def ball_detection(cam, debug=False):
    while True:
        start_time = time.time()
        # Read the image
        image, depth  = cam.read()
        # depth = cam.read()[1] 

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for yellow color
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])

        # Create a mask
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not debug:
            return contours
        # Draw circles around detected tennis balls
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # this value might need adjustment
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                cv2.circle(image, center, int(radius), (0, 255, 0), 2)

                # get the average depth of the ball based on the contour and depth image
                depth_ = depth[int(y)][int(x)]
                # convert to meters
                # get the depth scale of the camera
                depth_scale = cam.depth_scale
                depth_ = depth_ * depth_scale
                cv2.putText(image, str(depth_), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        fps = 1.0 / (time.time() - start_time)
        cv2.putText(image, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # Show the result
        if debug:
            cv2.imshow('Detected Tennis Balls', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            # get the average depth of the ball based on the contour and depth image
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    depth = depth[int(y)][int(x)]
                    

            return 

if __name__ == "__main__":
    cam = camera.RealSenseCamera(1280,720)
    ball_detection(cam, debug=True)    