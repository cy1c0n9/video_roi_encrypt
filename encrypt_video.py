# !/usr/bin/env python3

import redis
import json
import cv2

from attractor import HyperLu
from attractor import Protocol
from key import get_key_from_redis
from time import time
from vidgear.gears import WriteGear
import sys


def process(filename, video_folder="./videos/", dst_video_folder="./encrypted_video/"):
    """
    pre deliver_key data
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
    0                         500                                 1000
    :return:
    """
    redis_host = "localhost"
    redis_port = 6379
    redis_password = ""

    try:
        # The decode_responses flag here directs the client to convert the responses from Redis into Python strings
        r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)

        video_face_map = json.loads(r.hget("video_face_map", filename))
        time_face_map = json.loads(r.hget('time_face_map', filename))
        cap = cv2.VideoCapture(video_folder + filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        quad_fps = int((fps + 3) / 4)
        fps = int(fps + 0.5)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        video_writer = cv2.VideoWriter(dst_video_folder+filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        # define (Codec,CRF,preset) FFmpeg tweak parameters for writer
        # output_params = {"-vcodec": "mpeg4",
        #                  "-framerate": fps, "-crf": 5,
        #                  "-preset": "veryfast", "-r": fps,
        #                  "-qscale:v": 3}
        # video_writer = WriteGear(output_filename=dst_video_folder+filename,
        #                          compression_mode=True, logging=True, **output_params)

        # print("FPS: %d" % fps)
        # print("QUAD FPS: %d" % quad_fps)
        # setup the crypto system
        sender_dict = {}
        for fid in video_face_map.keys():
            key_4d = get_key_from_redis(fid)
            master = HyperLu.from_key(key_4d)
            sender = Protocol(master)
            sender.skip_first_n(key_4d.n)
            sender_dict[fid] = sender
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
        # Read until video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                start = time()
                # stamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                cur_frame = cap.get(1) - 1
                frame_width, frame_height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                # print(cur_frame)
                if cur_frame % fps < quad_fps:
                    msec_idx = int(cur_frame // fps * 1000)
                else:
                    msec_idx = int(cur_frame // quad_fps * 250)
                # print(msec_idx)

                for fid, idx in time_face_map[str(msec_idx)].items():
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
                    frame[y1:y2, x1:x2] = sender_dict[fid].encrypt(frame[y1:y2, x1:x2])

                # Display the resulting frame
                # cv2.imshow('Frame', frame)
                video_writer.write(frame)
                # end = time()
                # t = max(int(41 - 1000 * (end - start)), 1)
                # if t == 1:
                #     print(t)
                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break

        cap.release()
        # video_writer.close()
        video_writer.release()
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
    start_time = time()
    # process('103792584148574', "bbt_10s.mp4")
    process("bbt_10s.mp4")
    print("--- show in %s seconds ---" % (time() - start_time))
