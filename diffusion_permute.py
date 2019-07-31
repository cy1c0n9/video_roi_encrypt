from attractor import Lorenz
from attractor import Protocol
from attractor import Key
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import cv2
import math
import numpy as np
import time

# cap = cv2.VideoCapture(0)

video_folder = "./videos/"
filename = "103792584148574_1562618673.mp4"
cap = cv2.VideoCapture(video_folder + filename)
fps = cap.get(cv2.CAP_PROP_FPS)
fps = int(fps + 0.5)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# video_writer = cv2.VideoWriter('./tmp/tmp_encrypted.avi', cv2.VideoWriter_fourcc('F', 'F', 'V', '1'), fps, size)
video_writer = cv2.VideoWriter('./tmp/tmp_encrypted.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
video_writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 1.0)

key = Key.rand_init()
master = Lorenz.from_key(key)
slave = Lorenz.from_key(key)

sender = Protocol(master)
receiver = Protocol(slave)

sender.skip_first_n(key.n)
receiver.skip_first_n(key.n)

if not cap.isOpened():
    print("Error opening video stream or file")
# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        break
    start_time = time.time()
    # cv2.
    # imgsplit = cv2.split(frame)
    # cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb, frame)
    # for i in range(0, 90, 10):
    #     frame[i:i+10, i:i+10] = sender.encrypt(frame[i:i+10, i:i+10])
    #
    # for i in range(0, 90, 10):
    #     frame[i:i+10, i:i+10] = receiver.decrypt(frame[i:i+10, i:i+10])
    frame[10:100, 10:100] = sender.encrypt(frame[10:100, 10:100])
    # cv2.cvtColor(frame, cv2.COLOR_YCR_CB2BGR, frame)
    # frame[10:100, 10:100] = receiver.decrypt(frame[10:100, 10:100])
    # image = receiver.decrypt(encrypt_image)
    print("--- Encrypt %s seconds ---" % (time.time() - start_time))
    # cv2.imshow('frame', image)

    video_writer.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture


video_writer.release()
cap.release()
cv2.destroyAllWindows()

cap2 = cv2.VideoCapture('./tmp/tmp_encrypted.avi')
while cap2.isOpened():
    # Capture frame-by-frame
    ret, frame = cap2.read()
    if frame is None:
        break
    start_time = time.time()
    # encrypt_image = sender.encrypt(frame.copy())
    # for i in range(0, 90, 10):
    #     frame[i:i+10, i:i+10] = receiver.decrypt(frame[i:i+10, i:i+10])
    frame[10:100, 10:100] = receiver.decrypt(frame[10:100, 10:100])
    print("--- decrypt %s seconds ---" % (time.time() - start_time))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap2.release()
cv2.destroyAllWindows()
