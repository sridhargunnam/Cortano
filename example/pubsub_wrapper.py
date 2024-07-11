import multiprocessing as mp
from pubsub import pub

class PubSubWrapper:
    def __init__(self, queue_size=1):
        self.queues = {}
        self.queue_size = queue_size

    def create_queue(self, topic):
        if topic not in self.queues:
            self.queues[topic] = mp.Queue(self.queue_size)

    def publish(self, topic, data):
        self.create_queue(topic)
        queue = self.queues[topic]
        if queue.full():
            queue.get()  # Remove the oldest item
        queue.put(data)
        pub.sendMessage(topic, data=data)

    def subscribe(self, topic, callback):
        self.create_queue(topic)
        pub.subscribe(callback, topic)

    def get_queue(self, topic):
        self.create_queue(topic)
        return self.queues[topic]
