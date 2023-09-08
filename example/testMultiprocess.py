import multiprocessing
import random
import math
import time

def wait_for_n_ms(n=1):
    start_time = time.time()
    while (time.time() - start_time) * 1000 < 1*n:
        pass  # Wait until 1 ms has elapsed

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def producer(queue, num_productions):
    for _ in range(num_productions):
        print("inside producer, current queue size: ", queue.qsize())
        number = random.randint(100, 1000)
        queue.put(number)

def consumer(queue, num_productions):
    for _ in range(num_productions):
        print("inside consumer, current queue size: ", queue.qsize())
        number = queue.get()
        wait_for_n_ms(0)
        if is_prime(number):
            print(f"{number} is prime.")
        # else:
            # print(f"{number} is not prime.")

if __name__ == "__main__":
    num_productions = 10
    queue = multiprocessing.Queue(maxsize=10)

    producer_process = multiprocessing.Process(target=producer, args=(queue, num_productions))
    consumer_process = multiprocessing.Process(target=consumer, args=(queue, num_productions))

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
