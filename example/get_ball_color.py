import cv2
import numpy as np

def select_roi(image):
    """
    Allows the user to select a region of interest (ROI) on the image.
    Returns the coordinates of the selected rectangle.
    """
    r = cv2.selectROI("Image", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Image")
    return r

def average_hsv_in_roi(image, roi):
    """
    Converts the selected ROI to HSV color space and calculates the average HSV values.
    """
    x, y, w, h = roi
    roi = image[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    average_hsv = np.mean(hsv_roi, axis=(0, 1))
    return average_hsv

import camera 
def main():
    # Path to the image file
    # image_path = "/home/nvidia/wsp/clawbot/Cortano/no_contours_detected.png" #input("Enter the path to the image file: ")

    camRS = camera.RealSenseCamera(1280,720) 
    camera_paramsRS = camRS.getCameraIntrinsics()
    camDai = camera.DepthAICamera(1920, 1080)
    camera_paramsDai = camDai.getCameraIntrinsics() #1280,720)
    # Read the image
    image = camRS.read()[0]

    if image is None:
        print("Error: Image not found.")
        return

    # Select ROI
    roi = select_roi(image)

    # Compute average HSV values in the ROI
    average_hsv = average_hsv_in_roi(image, roi)
    print("Average HSV values in the selected region:", average_hsv)

if __name__ == "__main__":
    main()
