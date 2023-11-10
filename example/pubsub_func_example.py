import multiprocessing
import time
import random

def publisher(queue, control_queue, start_event, stop_event):
    """
    Function to act as a publisher which sends data to a queue.
    It can be started and stopped and its rate of publishing can be controlled.
    """
    rate = 1  # Default rate of publishing in seconds
    while not start_event.is_set():
        # Wait until the publisher is told to start
        time.sleep(0.1)

    print("Publisher started")
    while not stop_event.is_set():
        if not control_queue.empty():
            # Check if there is a new rate control message
            message = control_queue.get()
            if message == "STOP":
                break
            else:
                rate = message

        # Publish a random number to the queue
        num = random.randint(1, 100)
        queue.put(num)
        print(f"Published: {num}")
        time.sleep(rate)

    print("Publisher stopped")

def subscriber(queue, callback):
    """
    Function to act as a subscriber which receives data from a queue and uses a callback.
    """
    while True:
        num = queue.get()
        if num == "STOP":
            break
        callback(num)

def my_callback(data):
    """
    A simple callback function that processes data.
    """
    print(f"Subscriber received data: {data}")

if __name__ == "__main__":
    # Create a queue for publisher-subscriber communication
    queue = multiprocessing.Queue()

    # Create another queue for controlling the publisher
    control_queue = multiprocessing.Queue()

    # Create an event to control the start of the publisher
    start_event = multiprocessing.Event()

    # Create an event to signal the publisher to stop
    stop_event = multiprocessing.Event()

    # Create the publisher process
    publisher_process = multiprocessing.Process(target=publisher, args=(queue, control_queue, start_event, stop_event))

    # Create the subscriber process
    subscriber_process = multiprocessing.Process(target=subscriber, args=(queue, my_callback))

    # Start the processes
    publisher_process.start()
    subscriber_process.start()

    # Start the publisher
    start_event.set()

    # Change the publishing rate after 5 seconds
    time.sleep(5)
    control_queue.put(2)  # Change rate to every 2 seconds

    # Stop the publisher after 10 seconds
    time.sleep(10)
    stop_event.set()

    # Send stop signal to subscriber
    queue.put("STOP")

    # Wait for processes to finish
    publisher_process.join()
    subscriber_process.join()

    print("Program completed")
