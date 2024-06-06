# from time import sleep
# from multiprocessing import Process, set_start_method
# import sys


# print("This is from global scope", flush=True)


# # a custom function that blocks for a moment
# def task():
#     # block for a moment
#     sleep(1)
#     # display a message
#     print("This is from another process", flush=True)
#     sys.stdout.flush()


# def main_task():
#     # set the start method to 'spawn'
#     set_start_method("spawn", force=True)
#     # create a process
#     process = Process(target=task)
#     # run the process
#     process.start()
#     # wait for the process to finish
#     print("Waiting for the process...")
#     process.join()


# if __name__ == "__main__":
#     main_task()


from multiprocessing import Process
import os
from contextlib import redirect_stdout


class Worker(Process):
    def __init__(self, name, age, **kwargs):
        super().__init__()
        self.name = name
        self.age = age
        self.kwargs = kwargs

    def run(self):
        print(f"Worker: My name is {self.name} and my age is {self.age}", flush=True)
        print(f"Worker: My PID is {os.getpid()}", flush=True)
        print(f"Worker: My kwargs are: {self.kwargs}", flush=True)


if __name__ == "__main__":
    with open("log.txt", "w") as f:
        with redirect_stdout(f):
            print(f"This is from the main process: {os.getpid()}")
            p = Worker("John", 25, city="New York", country="USA")
            p.start()
            p.join()
