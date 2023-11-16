import cv2 
from PIL import Image
for i in range(10):
    try :
        cap = cv2.VideoCapture(i)
        Image.fromarray(cap.read()[1]).save("test.jpg")
        print("Try to open camera", i)
    except:
        pass