from time import sleep
from multiprocessing import Process, set_start_method


# a custom function that blocks for a moment
def task():
    # block for a moment
    sleep(1)
    # display a message
    print("This is from another process", flush=True)


# entry point
if __name__ == "__main__":
    set_start_method("spawn")
    # create a process
    process = Process(target=task)
    # run the process
    process.start()
    # wait for the process to finish
    print("Waiting for the process...")
    process.join()
