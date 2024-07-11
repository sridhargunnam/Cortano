#I want to codify a robotics pipeline in python. I want to implement a wrapper class for publisher subscriber model. I plan to have a robotics data processing. This is similar to how ROS implements things.  I have pubsub library in python in python called PyPubSub. But it is limited to single process. In this case I want to use multiprocessing so that everything can be parallelized.  In my example robotics pipeline there are multiple functions like camera, object detection, mapping, robotics control, april tag, etc.  example, camera process the data and gives out the color, depth, and camera params. The next function could be april tag detecter, object detection, which will need the frames from camera function. The camera function needs to likely two separate queues one for each april tag detector, and another for object detection such that both can happen hand in hand. I plan to use multiprocessing for each of these functions. The other functions will be detecting the location of ball w.r.t to the robot. This location mapper function will the object co-ordinates from object detection function(which will send the detections using a queue(publisher), the camera also publishes the depth frame only to a queue, which can again be published. The location mapper will be the subscribe to detections, and depth. It will need to put out the relation location to a queue, which will be consumed by the robotics control function. Essentially first create a pub sub wrapper and also automatically create the queue's when published. Please think in a step by step process. 

import multiprocessing as mp
from collections import defaultdict
from typing import Callable, Any

class PubSub:
    def __init__(self):
        self.queues = defaultdict(mp.Queue)
        self.subscribers = defaultdict(list)

    def publish(self, topic: str, message: Any):
        for queue in self.subscribers[topic]:
            queue.put(message)

    def subscribe(self, topic: str) -> mp.Queue:
        queue = mp.Queue()
        self.subscribers[topic].append(queue)
        return queue

    def create_queue(self, topic: str) -> mp.Queue:
        return self.queues[topic]

def camera_process(pubsub: PubSub):
    import time
    while True:
        # Simulate capturing data
        color_frame = "color_frame_data"
        depth_frame = "depth_frame_data"
        camera_params = "camera_params"

        # Publish the data to respective topics
        pubsub.publish("camera/color", color_frame)
        pubsub.publish("camera/depth", depth_frame)
        pubsub.publish("camera/params", camera_params)

        time.sleep(1)  # Simulate time delay in capturing data

def april_tag_detector(pubsub: PubSub):
    color_queue = pubsub.subscribe("camera/color")

    while True:
        color_frame = color_queue.get()  # Blocking call to wait for the next frame

        # Simulate April Tag detection
        tag_detections = f"detected_tags_in_{color_frame}"

        pubsub.publish("april_tag/detections", tag_detections)

def object_detection(pubsub: PubSub):
    color_queue = pubsub.subscribe("camera/color")

    while True:
        color_frame = color_queue.get()  # Blocking call to wait for the next frame

        # Simulate object detection
        object_detections = f"detected_objects_in_{color_frame}"

        pubsub.publish("object_detection/detections", object_detections)

def location_mapper(pubsub: PubSub):
    object_queue = pubsub.subscribe("object_detection/detections")
    depth_queue = pubsub.subscribe("camera/depth")

    while True:
        object_detections = object_queue.get()  # Blocking call to wait for the next detection
        depth_frame = depth_queue.get()         # Blocking call to wait for the next depth frame

        # Simulate computing the location of objects
        relative_location = f"location_of_{object_detections}_using_{depth_frame}"

        pubsub.publish("location_mapper/locations", relative_location)

def robotics_control(pubsub: PubSub):
    location_queue = pubsub.subscribe("location_mapper/locations")

    while True:
        relative_location = location_queue.get()  # Blocking call to wait for the next location

        # Simulate robot control based on the relative location
        control_command = f"control_command_based_on_{relative_location}"

        pubsub.publish("robotics_control/commands", control_command)
if __name__ == "__main__":
    pubsub = PubSub()

    processes = [
        mp.Process(target=camera_process, args=(pubsub,)),
        mp.Process(target=april_tag_detector, args=(pubsub,)),
        mp.Process(target=object_detection, args=(pubsub,)),
        mp.Process(target=location_mapper, args=(pubsub,)),
        mp.Process(target=robotics_control, args=(pubsub,))
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
