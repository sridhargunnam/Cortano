import cv2
import numpy as np
import time 
import camera
import multiprocessing as mp

def distance(p1, p2):
    """Calculate distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def run_object_detection_mp(rs_queue_ocv, rs_ocv_ball_detection_queue, depth_scale=1.0):
    while True:
        if rs_queue_ocv.empty():
            continue
        image = rs_queue_ocv.get()
        if image is None:
            break
        rs_ocv_ball_detection_queue.put(ball_detection(image))

def ball_detection(image, debug=False, min_distance=30): # Added min_distance parameter
    # while True:
        start_time = time.time()
        # Read the image
        # image, depth  = cam.read()
        # depth = cam.read()[1] 
        # depth_3d = cam.depth2rgb(depth)

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for yellow color
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])

        # Create a mask
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > 30]

        total_contours = len(contours)
        centroids = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            # else:
            #     cx, cy = 0, 0
                centroids.append((cx, cy))
        
        # Filtering contours based on centroid distance
        unique_contours = []
        for i, c1 in enumerate(centroids):
            if all(distance(c1, c2) >= min_distance for j, c2 in enumerate(centroids) if i != j):
                unique_contours.append(contours[i])

        if not debug:
            result = []
            for i in range(len(contours)):
                (center_x, center_y), radius = cv2.minEnclosingCircle(contours[i])
                result.append ({'center_x': center_x, 'center_y': center_y, 'width': radius, 'height': radius})
            return result
    
        print("total_contours = ", total_contours, "unique_contours = ", len(unique_contours))
        # Draw circles around detected tennis balls
        for contour in unique_contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            cv2.circle(image, center, int(radius), (0, 255, 0), 2)
            # # get the average depth of the ball based on the contour and depth image
            # depth_ = depth[int(y)][int(x)]
            # # convert to meters
            # # get the depth scale of the camera
            # depth_scale = cam.depth_scale
            # depth_ = depth_ * depth_scale
            # cv2.putText(image, str(depth_), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # cv2.putText(depth_3d, str(depth_), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        #displayed fps is incorrect
        # end_time = time.time()
        # fps = 1.0 / (end_time - start_time)
        # start_time = end_time

        # cv2.putText(image, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # # cv2.putText(depth_3d, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # Show the result
        if debug:
            # cv2.imshow('Depth', depth_3d)
            downscale = 1 #to view 
            cv2.imshow('Color', cv2.resize(image, (0,0), fx=downscale, fy=downscale))
            # cv2.imshow('Color', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                exit(0)
                

# def ball_detection(cam, debug=False):
#     while True:
#         start_time = time.time()
#         # Read the image
#         image, depth  = cam.read()
#         # depth = cam.read()[1] 
#         depth_3d = cam.depth2rgb(depth)

#         # Convert to HSV
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#         # Define range for yellow color
#         lower_yellow = np.array([20, 100, 100])
#         upper_yellow = np.array([40, 255, 255])

#         # Create a mask
#         mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

#         # Find contours in the mask
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not debug:
#             return contours
#         # Draw circles around detected tennis balls
#         for contour in contours:
#             if cv2.contourArea(contour) > 100:  # this value might need adjustment
#                 (x, y), radius = cv2.minEnclosingCircle(contour)
#                 center = (int(x), int(y))
#                 cv2.circle(image, center, int(radius), (0, 255, 0), 2)

#                 # get the average depth of the ball based on the contour and depth image
#                 depth_ = depth[int(y)][int(x)]
#                 # convert to meters
#                 # get the depth scale of the camera
#                 depth_scale = cam.depth_scale
#                 depth_ = depth_ * depth_scale
#                 cv2.putText(image, str(depth_), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#                 cv2.putText(depth_3d, str(depth_), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


#         fps = 1.0 / (time.time() - start_time)
#         cv2.putText(image, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#         cv2.putText(depth_3d, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#         # Show the result
#         if debug:
#             cv2.imshow('Depth', depth_3d)
#             cv2.imshow('Color', image)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 cv2.destroyAllWindows()
#                 break

if __name__ == "__main__":
    # cam = camera.RealSenseCamera(1280,720)
    # cam = camera.DepthAICamera(1920,1080)
    cam = camera.DepthAICamera(640,360)
    start_time = time.time()
    while True:
        image = cam.read()[0]
        if image is None:
            continue
        else:
            ball_detection(cam.read()[0], debug=True)    
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            start_time = end_time
            print("FPS: {:.2f}".format(fps))