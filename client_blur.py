# !/usr/bin/env python3

import cv2
from time import time


def play(filename):
    """
    pre deliver_key data
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
    0                         500                                 1000
    :return:
    """
    try:
        video_folder = "./blur/"
        cap = cv2.VideoCapture(video_folder + filename)
        fps = cap.get(cv2.CAP_PROP_FPS)

        fps = int(fps + 0.5)
        interval = int((1 / fps) * 1000)

        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
        # Read until video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                start = time()
                # Display the resulting frame
                cv2.imshow('Frame', frame)
                end = time()
                t = max(int(interval - 1000 * (end - start)), 1)
                # print(interval)
                # Press Q on keyboard to  exit
                if cv2.waitKey(t) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print('error: ')
        print(e)


if __name__ == '__main__':

    start_time = time()
    # process('103792584148574', "bbt_10s.mp4")
    play("bbt_10s.mp4")
    print("--- show in %s seconds ---" % (time() - start_time))
