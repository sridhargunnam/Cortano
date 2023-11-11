import pyrealsense2 as rs
import numpy as np
import cv2
# Function to load mask
import config 
config = config.Config()

contour_points = []  # Global variable to store points of the contour

# Mouse callback function for drawing
def draw_contour(event, x, y, flags, param):
    global contour_points  # Declare contour_points as a global variable
    if event == cv2.EVENT_LBUTTONDOWN:
        contour_points.append((x, y))


def create_mask(width, height):
    # Initialize and start RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        # Skip first 100 frames for camera warm-up
        for _ in range(100):
            pipeline.wait_for_frames()

        # Capture one frame to draw the mask on
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Could not capture image from camera")

        # Convert image to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Setup for drawing mask
        mask = np.zeros(color_image.shape[:2], dtype=np.uint8)  # Mask is a single channel image
        cv2.namedWindow('Draw Mask')
        cv2.setMouseCallback('Draw Mask', draw_contour)

        while True:
            temp_image = color_image.copy()
            if len(contour_points) > 1:
                cv2.polylines(temp_image, [np.array(contour_points)], False, (0, 255, 0), 2)
            cv2.imshow('Draw Mask', temp_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # Press 's' to save and exit
                if len(contour_points) > 2:
                    cv2.fillPoly(mask, [np.array(contour_points)], (255))
                cv2.imwrite('mask.png', mask)
                break
            elif key == ord('q'):  # Press 'q' to exit without saving
                break

        cv2.destroyAllWindows()

    finally:
        pipeline.stop()


    
def apply_mask_to_feed(width, height):
    # Load the saved mask
    mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError("Mask image not found")

    assert mask.shape[:2] == (height, width), "Mask size does not match image size"


    # Initialize and start RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        # Skip first 100 frames for camera warm-up
        for _ in range(100):
            pipeline.wait_for_frames()

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert image to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Apply the mask
            masked_image = cv2.bitwise_and(color_image, color_image, mask=mask)

            # Display the result
            cv2.imshow('Masked Feed', masked_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if config.CREATE_MASK is True:
    create_mask(1280, 720)
    apply_mask_to_feed(1280, 720)


def load_mask(mask_path = config.MASK_PATH):
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Function to check if point is inside mask
def is_point_in_mask(point, mask):
    x, y = point
    return mask[y, x] > 0