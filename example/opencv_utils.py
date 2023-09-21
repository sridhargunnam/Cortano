import camera
import cv2
import numpy as np
# import matplotlib.pyplot as plt

# getCircles takes a color image and returns a masks and boxes of the circles in the image, using opencv
def getCircles(color, minRadiusUser=0, maxRadiusUser=0):
    # convert to grayscale
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    # blur the image
    gray = cv2.medianBlur(gray, 5)
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=minRadiusUser, maxRadius=maxRadiusUser)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        # round the floating point values to integers
        circles = np.round(circles[0, :]).astype("int")
        # return the (x, y) coordinates and radius of the circles
        # return circles
        # return the circles
        return circles
    else:
        return None

def getBallReferenceColor(ball_type):
    if ball_type == "pingPong":
        # orange = (255, 165, 0)
        # In the RGB format, the color orange is represented by the following values: Red=255, Green=165 and Blue=0.
        lower_orange = np.array([180,70,10])
        upper_orange = np.array([255,130,100])

        # Define lower and upper threshold values for orange in hsv
        lower_orange = np.array([0, 150, 190])  # Adjust the lower values as needed
        upper_orange = np.array([20, 230, 255])  # Adjust the upper values as needed
        # mask = cv2.inRange(hsv,(10, 100, 20), (25, 255, 255) )

        reference_color = (lower_orange, upper_orange)
    else:
        reference_color = None
    return reference_color

#get circles using the second method, that uses a reference color for circles
save_dir = "/home/nvidia/wsp/clawbot/Cortano/example/debug_data/"
def getCircles2(color, reference_color, minRadiusUser=50, maxRadiusUser=65):
    # convert to hsv
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    # get the mask
    mask = cv2.inRange(hsv, reference_color[0], reference_color[1])
    # blur the mask
    mask = cv2.medianBlur(mask, 5)
    # write mask, hsv, color to file
    cv2.imwrite(save_dir + "mask.png", mask)
    cv2.imwrite(save_dir + "hsv.png", hsv)
    cv2.imwrite(save_dir + "color.png", color)
    #show the mask on the image
    # cv2.imshow("mask", mask)
    # cv2.waitKey(5)
    # detect circles in the image
    #FIXME get min and max radius based on the type of the ball
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=minRadiusUser, maxRadius=maxRadiusUser)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        # round the floating point values to integers
        circles = np.round(circles[0, :]).astype("int")
        # return the (x, y) coordinates and radius of the circles
        # return circles
        # return the circles
        return circles
    else:
        # print("No circles found in function getCircles2")
        return None


def drawCircles(color, circles):
    # draw the circles
    for (x, y, r) in circles:
        cv2.circle(color, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(color, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    cv2.imshow("output", np.hstack([color]))
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        exit(0)
        # cv2.waitKey(0)

def getCircleColorRangePerChannel(color, circle, channel, thresholdHist=True):
    x,y,r = circle
    channel = color[:,:,channel]
    # calculate the histogram of each color channel for the pixels that are inside the circle 
    # Manually calculate the histogram of each color channel
    channel_hist = np.zeros(256)
    for i in range(x-r, x+r):
        for j in range(y-r, y+r):
            if (i-x)**2 + (j-y)**2 <= r**2:
                try:
                    channel_hist[channel[j,i]] += 1
                except:
                    print("Exception at i,j", i,j)
                    pass
    #set threshold for the histogram
    if thresholdHist:
        threshold = 0.1 * np.max(channel_hist)
        for i in range(len(channel_hist)):
            if channel_hist[i] < threshold:
                channel_hist[i] = 0
        # channel_hist = channel_hist[channel_hist > threshold]
    
    return channel_hist
# given a color image and an circle i.e x,y coordinate and the radius in terms of pixels, get the range of color inside the circle 
# Get the histogram of each color channels and plot them
def getCircleColorRange(color, circle):
    # define getCircleColorRangePerChannel function
    # get the histogram of each color channel
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    blue_channel_hist = getCircleColorRangePerChannel(hsv, circle, 0)
    green_channel_hist = getCircleColorRangePerChannel(hsv, circle, 1)
    red_channel_hist = getCircleColorRangePerChannel(hsv, circle, 2)

    # print the range where pixel intensity values that satisfy the threshold are found    
    print("Blue channel values = ", np.where(blue_channel_hist > min(blue_channel_hist)))
    print("Green channel values = ", np.where(green_channel_hist > min(green_channel_hist)))
    print("Red channel values = ", np.where(red_channel_hist > min(red_channel_hist)))

    # print("Blue channel min, max = ", np.min(blue_channel_hist), np.max(blue_channel_hist))
    # print("Green channel min, max = ", np.min(green_channel_hist), np.max(green_channel_hist))
    # print("Red channel min, max = ", np.min(red_channel_hist), np.max(red_channel_hist))

    # plot the histogram of each color channel
    # plt.figure()
    # plt.subplot(3,1,1)
    # plt.plot(blue_channel_hist)
    # plt.title("Blue channel histogram")
    # plt.subplot(3,1,2)
    # plt.plot(green_channel_hist)
    # plt.title("Green channel histogram")
    # plt.subplot(3,1,3)
    # plt.plot(red_channel_hist)
    # plt.title("Red channel histogram")
    #FIXME 1 plt and imshow doesn't work together
    # plt.show()
    # plt.close()

          
def detectCirclesAlgo1():
    cam = camera.RealSenseCamera(1280,720) 
    # skip first 10 frames as they are of low exposure
    max_tries = 100
    # for i in range(10):
    #     color, depth = cam.read()
    while True:
        try:
            color, depth = cam.read()
            circles = getCircles(color) #, 5, 30)
            if circles is not None:
                if len(circles) > 5:
                    numberOfCirclesToDraw = 5
                else:
                    numberOfCirclesToDraw = len(circles)    
                for (x, y, r) in circles[:numberOfCirclesToDraw+1]:
                    print("x,y,r = ", x,y,r)
                    cv2.circle(color, (x, y), r, (0, 255, 0), 4)
                    cv2.rectangle(color, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        except:
            if max_tries > 0:
                print("could not get circles")
                max_tries -= 1
            else:
                exit(0)
      
        cv2.imshow("output", np.hstack([color]))
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            exit(0)
            
def blobDetector():
        # Read the image
        # cam = camera.RealSenseCamera(1280,720) 
        im = cv2.imread("/home/nvidia/Downloads/photos/PXL_20230919_032555846.jpg", 0)
        # Setup BlobDetector
        detector = cv2.SimpleBlobDetector_create()
        params = cv2.SimpleBlobDetector_Params()
            
        # Filter by Area.
        # params.filterByArea = True
        # params.minArea = 30
        # params.maxArea = 40000
            
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.9
        
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.87
            
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.8

        # Distance Between Blobs
        # params.minDistBetweenBlobs = 200
            
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)

        
        overlay = im.copy()

        keypoints = detector.detect(im)
        for k in keypoints:
            cv2.circle(overlay, (int(k.pt[0]), int(k.pt[1])), int(k.size/2), (0, 0, 255), -1)
            cv2.line(overlay, (int(k.pt[0])-20, int(k.pt[1])), (int(k.pt[0])+20, int(k.pt[1])), (0,0,0), 3)
            cv2.line(overlay, (int(k.pt[0]), int(k.pt[1])-20), (int(k.pt[0]), int(k.pt[1])+20), (0,0,0), 3)

        opacity = 0.5
        cv2.addWeighted(overlay, opacity, im, 1 - opacity, 0, im)

        # Uncomment to resize to fit output window if needed
        im = cv2.resize(im, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

        cv2.imshow("Output", im)

        if cv2.waitKey(1) == 27:
           cv2.destroyAllWindows()
           exit(0)

        # while True:
            # image = cam.read()[0]
            # # Convert the image to grayscale
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # # Initialize the blob detector
            # params = cv2.SimpleBlobDetector_Params()

            # # Set blob detection parameters
            # params.filterByColor = True
            # params.blobColor = 255
            # params.minThreshold = 127    # Minimum threshold value to consider a pixel as part of a blob
            # params.maxThreshold = 200   # Maximum threshold value
            # params.filterByArea = True  # Filter blobs by area
            # params.minArea = 30       # Minimum blob area in pixels
            # params.filterByCircularity = True  # Filter blobs by circularity
            # params.minCircularity = 0.8      # Minimum circularity
            # params.filterByConvexity = True   # Filter blobs by convexity
            # params.minConvexity = 0.87       # Minimum convexity
            # params.filterByInertia = True    # Filter blobs by inertia
            # params.minInertiaRatio = 0.01   # Minimum inertia ratio

            # # Create the blob detector with the specified parameters
            # detector = cv2.SimpleBlobDetector_create(params)

            # # Detect blobs in the image
            # keypoints = detector.detect(gray)

            # # Draw detected blobs on the image
            # result_image = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
            #                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # # Display the original image with detected blobs
            # cv2.imshow('Blob Detection', result_image)
            # if cv2.waitKey(1) == 27:
            #     cv2.destroyAllWindows()
            #     exit(0)

            # Read image
            


if __name__ == "__main__":
    detectCirclesAlgo1()
    # while True:
    #     blobDetector()

  