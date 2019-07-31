from attractor import Lorenz
from attractor import Key
from attractor import Protocol
import numpy as np
import cv2
from time import time

video_folder = "./videos/"
filename = "103792584148574_1562618673.mp4"
cap = cv2.VideoCapture(video_folder + filename)
fps = cap.get(cv2.CAP_PROP_FPS)
fps = int(fps + 0.5)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

video_writer = cv2.VideoWriter('./tmp/tmp_encrypted.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)
video_writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 1.0)
# key = np.random.uniform(-10, 10), np.random.uniform(-15, 15), np.random.uniform(0, 40)
key_l = [1.1381133082873731, 12.990637231862848, 0.8822348261655519, 1000]

# init encoder
key_m = Key.from_list(key_l)
master = Lorenz.from_key(key_m)
sender = Protocol(master)
sender.skip_first_n(key_m.n)

# save key
# print(str(key_m))
key_str = str(key_m)

# init decoder
key_s = Key.from_str(key_str)
slave = Lorenz.from_key(key_s)
receiver = Protocol(slave)
receiver.skip_first_n(key_s.n)
print(master)
print(slave)

# print(key_l[0])
# print(key_l[0] * 10000)

if not cap.isOpened():
    print("Error opening video stream or file")
# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if frame is None:
        break
    start_time = time()
    # cv2.
    # imgsplit = cv2.split(frame)
    # cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb, frame)
    # for i in range(0, 90, 10):
    #     frame[i:i+10, i:i+10] = sender.encrypt(frame[i:i+10, i:i+10])
    #
    # for i in range(0, 90, 10):
    #     frame[i:i+10, i:i+10] = receiver.decrypt(frame[i:i+10, i:i+10])
    frame[10:100, 10:90] = sender.encrypt(frame[10:100, 10:90])
    # cv2.cvtColor(frame, cv2.COLOR_YCR_CB2BGR, frame)
    # frame[10:100, 10:100] = receiver.decrypt(frame[10:100, 10:100])
    # image = receiver.decrypt(encrypt_image)
    print("--- Encrypt %s seconds ---" % (time() - start_time))
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
    start_time = time()
    # encrypt_image = sender.encrypt(frame.copy())
    # for i in range(0, 90, 10):
    #     frame[i:i+10, i:i+10] = receiver.decrypt(frame[i:i+10, i:i+10])
    frame[10:100, 10:90] = receiver.decrypt(frame[10:100, 10:90])
    print("--- decrypt %s seconds ---" % (time() - start_time))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap2.release()
cv2.destroyAllWindows()
