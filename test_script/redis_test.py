# !/usr/bin/env python3

import redis
import json
import cv2

redis_host = "localhost"
redis_port = 6379
redis_password = ""
filename = "103792584148574_1561999275.mp4"


def process():
    """
    pre deliver_key data

    :return:
    """
    try:
        # The decode_responses flag here directs the client to convert the responses from Redis into Python strings
        r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)

        video_face_map = json.loads(r.hget("video_face_map", filename))
        time_face_map = json.loads(r.hget('time_face_map', filename))
        print(video_face_map)
        print(time_face_map)
        cap = cv2.VideoCapture("videos/" + filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
        calc_timestamps = [0.0]
        print(fps)
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
        # Read until video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # Display the resulting frame
                cv2.imshow('Frame', frame)
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                calc_timestamps.append(calc_timestamps[-1] + 1000 / fps)

            # Break the loop
            else:
                break

        cap.release()

        for i, (ts, cts) in enumerate(zip(timestamps, calc_timestamps)):
            print('Frame %d difference:' % i, abs(ts - cts))
        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)


if __name__ == '__main__':
    process()
