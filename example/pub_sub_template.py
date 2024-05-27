import multiprocessing as mp
from pubsub import pub
import time
import random

class PubSubWrapper:
    def __init__(self):
        self.queues = {}

    def create_queue(self, topic):
        if topic not in self.queues:
            self.queues[topic] = mp.Queue()

    def publish(self, topic, data):
        self.create_queue(topic)
        self.queues[topic].put(data)
        pub.sendMessage(topic, data=data)

    def subscribe(self, topic, callback):
        self.create_queue(topic)
        pub.subscribe(callback, topic)

    def get_queue(self, topic):
        self.create_queue(topic)
        return self.queues[topic]

def camera_process(pubsub):
    while True:
        color_frame = random.randint(0, 255)
        depth_frame = random.randint(0, 255)
        camera_params = {"param1": random.random()}
        pubsub.publish('camera/color', color_frame)
        pubsub.publish('camera/depth', depth_frame)
        pubsub.publish('camera/params', camera_params)
        print(f"Camera: Color={color_frame}, Depth={depth_frame}, Params={camera_params}")
        # time.sleep(1)

def april_tag_detector(pubsub, camera_queue):
    while True:
        color_frame = camera_queue.get()
        detections = f"AprilTag Detected on frame {color_frame}"
        pubsub.publish('april_tag/detections', detections)
        print(detections)
        # time.sleep(1)

def object_detection(pubsub, camera_queue):
    while True:
        color_frame = camera_queue.get()
        detections = f"Object Detected on frame {color_frame}"
        pubsub.publish('object/detections', detections)
        print(detections)
        # time.sleep(1)

def location_mapper(pubsub, object_queue, depth_queue):
    while True:
        object_detections = object_queue.get()
        depth_frame = depth_queue.get()
        location = f"Location calculated using {object_detections} and depth {depth_frame}"
        pubsub.publish('location', location)
        print(location)
        # time.sleep(1)

def robotics_control(pubsub, location_queue):
    while True:
        location = location_queue.get()
        control_commands = f"Control commands for {location}"
        print(control_commands)
        # time.sleep(1)

def start_pipeline():
    pubsub = PubSubWrapper()

    camera_queue_color = pubsub.get_queue('camera/color')
    camera_queue_depth = pubsub.get_queue('camera/depth')
    object_detection_queue = pubsub.get_queue('object/detections')
    april_tag_queue = pubsub.get_queue('april_tag/detections')
    location_queue = pubsub.get_queue('location')

    camera_proc = mp.Process(target=camera_process, args=(pubsub,))
    april_tag_proc = mp.Process(target=april_tag_detector, args=(pubsub, camera_queue_color))
    object_detection_proc = mp.Process(target=object_detection, args=(pubsub, camera_queue_color))
    location_mapper_proc = mp.Process(target=location_mapper, args=(pubsub, object_detection_queue, camera_queue_depth))
    robotics_control_proc = mp.Process(target=robotics_control, args=(pubsub, location_queue))

    camera_proc.start()
    april_tag_proc.start()
    object_detection_proc.start()
    location_mapper_proc.start()
    robotics_control_proc.start()

    camera_proc.join()
    april_tag_proc.join()
    object_detection_proc.join()
    location_mapper_proc.join()
    robotics_control_proc.join()

if __name__ == "__main__":
    start_pipeline()
