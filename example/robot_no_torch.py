import camera
from PIL import Image
import numpy as np
import cv2

fx = 460.92495728
fy = 460.85058594
cx = 315.10949707
cy = 176.72598267
h, w = (360, 640)
U = np.tile(np.arange(w).reshape((1, w)), (h, 1))
V = np.tile(np.arange(h).reshape((h, 1)), (1, w))
U = (U - cx) / fx
V = (V - cy) / fy

def get_XYZ(depth_image):
    Z = depth_image
    X = U * Z
    Y = V * Z
    # formatting magic
    XYZ = np.concatenate((
        X.reshape((-1, 1)),
        Y.reshape((-1, 1)),
        Z.reshape((-1, 1))
    ), axis=-1)
    return XYZ
# getCircles takes a color image and returns a masks and boxes of the circles in the image, using opencv
def getCircles(color):
    # convert to grayscale
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    # blur the image
    gray = cv2.medianBlur(gray, 5)
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        # round the floating point values to integers
        circles = np.round(circles[0, :]).astype("int")
        # return the (x, y) coordinates and radius of the circles
        # return circles
        # draw the circles
        # for (x, y, r) in circles:
        #     cv2.circle(color, (x, y), r, (0, 255, 0), 4)
        #     cv2.rectangle(color, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        #     cv2.imshow("output", np.hstack([color]))
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # return the circles
        return circles
    else:
        return None

    
if __name__ == "__main__":
    # parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()
    debug_mode = args.debug

    # robot = RemoteInterface("...")
    cam = camera.RealSenseCamera() # because this is a locally run camera, but you don't need


    #while robot.running():
    while True:
        # color, depth, sensors = robot.read()
        color, depth = cam.read()

        output = getCircles(color)
        output = {l: output[l].to('cpu').numpy() for l in output}

        object_index = 37
        indeces_found = [i for i in range(len(output["labels"])) if \
            (output["labels"][i] == object_index and output["scores"][i] > 0)]
        scores = output["scores"][indeces_found] # between 0 and 1, 1 being most confident
        labels = output["labels"][indeces_found] # integer id of the object found, refer to COCO classes
        masks  = output["masks"] [indeces_found] # mask in the image (N, 1, height, width)
        boxes  = output["boxes"] [indeces_found] # bounding boxes, not really needed

        # get a single mask's centered XYZ coordinates, ball's location
        single_mask = np.zeros((360, 640), dtype=np.uint8)
        if len(masks) > 0:
            single_mask = masks[0].reshape((360, 640))
        ball_depth = depth * (single_mask > 0)
        xyz = get_XYZ(ball_depth)
        num_pixels = np.sum(ball_depth > 0)
        if num_pixels > 0:
            average_xyz = np.sum(xyz, axis=0) / num_pixels
        
        if debug_mode:
            if num_pixels > 0:
                print(average_xyz)
            #draw the bounding boxes and masks  on the image
            
            # run this in debug mode only once
            exit_flag = False
            if not exit_flag:    
                for i in range(len(indeces_found)):
                    mask = masks[i].reshape((360, 640))
                    color[mask > 0] = (0, 255, 0)
                    cv2.rectangle(color, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])), (0, 0, 255), 2)
                    cv2.putText(color, str(labels[i]), (int(boxes[i][0]), int(boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("color", color)
                    # cv2.waitKey(1)
                    if cv2.waitKey(0) in [27, ord('q'), ord('Q')]:
                        exit_flag = True
                        debug_mode = False
                        break
                cv2.destroyAllWindows()
            """
            # calculate the average fps = ~1000 fps on desktop(4060+amd 3600)
            import time
            start = time.time()
            for i in range(100):
                with torch.no_grad():
                    output = model(input_batch)[0]
                    end = time.time()
                    print("FPS:", 100 / (end - start))
                    start = end
            
            # calculate the average inference time = ~100 fps on desktop(4060)
            start = time.time()
            for i in range(100):
                with torch.no_grad():
                    output = model(input_batch)[0]
                    end = time.time()
                    print("Inference time:", (end - start) * 1000, "ms")
                    start = end
            """




        


