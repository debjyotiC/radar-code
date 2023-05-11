import threading
import os


def function1():
    # code for the first thread
    os.system(
        "C:\\Users\\Deb\\anaconda3\\envs\\radar-code\\python.exe C:\\Users\\Deb\\PycharmProjects\\radar-code\\radar_data_web_render.py")
    print("This is running in the first thread")


def function2():
    os.system(
        "C:\\Users\\Deb\\anaconda3\\envs\\radar-code\\python.exe C:\\Users\\Deb\\PycharmProjects\\radar-code\\real_time_classifier.py")
    print("This is running in the second thread")


# create the first thread
thread1 = threading.Thread(target=function1)

# create the second thread
thread2 = threading.Thread(target=function2)

# start the threads
thread1.start()
thread2.start()

# wait for the threads to finish
thread1.join()
thread2.join()

# do some other work here in the main thread
print("Both threads have finished running")
