# !/usr/bin/env python3

import json
import cv2

from attractor import HyperLu
from attractor import Protocol
from key import Key4D
from time import time
import sys


def play(filename, json_data, video_folder="./encrypted_video/"):
    """
    pre deliver_key data
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
    0                         500                                 1000
    :return:
    """
    try:
        # The decode_responses flag here directs the client to convert the responses from Redis into Python strings

        video_face_map = json_data['video_face_map']
        # print(video_face_map)
        time_face_map = json_data['time_face_map']
        # print(time_face_map)
        face_key = json_data['face_key']
        # print(face_key)
        cap = cv2.VideoCapture(video_folder + filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        quad_fps = int((fps + 3) / 4)
        fps = int(fps + 0.5)
        interval = int((1 / fps) * 1000)

        # setup the crypto system
        receiver_dict = {}
        for fid in video_face_map.keys():
            key_4d = Key4D.from_str(face_key[fid])
            slaver = HyperLu.from_key(key_4d)
            receiver = Protocol(slaver)
            receiver.skip_first_n(key_4d.n)
            receiver_dict[fid] = receiver

        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
        # Read until video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                start = time()
                cur_frame = cap.get(1) - 1
                if cur_frame == 59:
                    a = 1
                frame_width, frame_height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if cur_frame % fps < quad_fps:
                    msec_idx = int(cur_frame // fps * 1000)
                else:
                    msec_idx = int(cur_frame // quad_fps * 250)
                for fid in time_face_map[str(msec_idx)]:
                    pos = video_face_map[fid][str(msec_idx)]
                    # x1 = int(pos['x'] - 1 / 12 * pos['width'])
                    # y1 = int(pos['y'] - 1 / 12 * pos['height'])
                    # x2 = int(pos['x'] + 1.2 * pos['width'])
                    # y2 = int(pos['y'] + 1.2 * pos['height'])
                    # x1, y1 = max(x1, 0), max(y1, 0)
                    # x2, y2 = min(x2, int(frame_width)), min(y2, int(frame_height))
                    x1 = int(pos['x'])
                    y1 = int(pos['y'])
                    x2 = int(pos['x'] + pos['width'])
                    y2 = int(pos['y'] + pos['height'])
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, int(frame_width)), min(y2, int(frame_height))
                    frame[y1:y2, x1:x2] = receiver_dict[fid].decrypt(frame[y1:y2, x1:x2])

                # Display the resulting frame
                cv2.imshow('Frame', frame)
                end = time()
                t = max(int(interval - 1 - 1000 * (end - start)), 1)
                if t == 1:
                    print(t)
                if cv2.waitKey(t) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print('error: system argument')
    #     raise AttributeError
    # else:
    #     file_name = sys.argv[1]
    # process(filename=file_name)
    with open('./json/' + "bbt_10s.mp4" + '.json', 'r') as f:
        data_store = json.load(f)
    start_time = time()
    # process('103792584148574', "bbt_10s.mp4")
    play("bbt_10s.mp4", data_store)
    print("--- show in %s seconds ---" % (time() - start_time))
