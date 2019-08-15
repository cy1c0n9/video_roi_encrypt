from key import get_key_from_redis
from key import generate_new_key

import redis
import json
import cv2
from attractor import HyperLu
from attractor import Protocol
from time import time


def key_revoke(faceid, filename, video_folder="./encrypted_video/"):
    # decrypt the video with old key
    redis_host = "localhost"
    redis_port = 6379
    redis_password = ""
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
    video_writer = cv2.VideoWriter("./tmp/" + filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    # setup the crypto system
    key_4d_old = get_key_from_redis(faceid)
    slaver = HyperLu.from_key(key_4d_old)
    receiver = Protocol(slaver)
    receiver.skip_first_n(key_4d_old.n)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            cur_frame = cap.get(1) - 1
            frame_width, frame_height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if cur_frame % fps < quad_fps:
                msec_idx = int(cur_frame // fps * 1000)
            else:
                msec_idx = int(cur_frame // quad_fps * 250)
            if faceid in time_face_map[str(msec_idx)]:
                pos = video_face_map[faceid][str(msec_idx)]
                x1 = int(pos['x'])
                y1 = int(pos['y'])
                x2 = int(pos['x'] + pos['width'])
                y2 = int(pos['y'] + pos['height'])
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, int(frame_width)), min(y2, int(frame_height))
                frame[y1:y2, x1:x2] = receiver.decrypt(frame[y1:y2, x1:x2])

            video_writer.write(frame)
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    video_writer.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    # generate new key
    key_4d_new = generate_new_key(faceid)

    # encrypt the face using new key
    cap = cv2.VideoCapture('./tmp/' + filename)
    video_writer = cv2.VideoWriter(video_folder + filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    # setup the crypto system
    master = HyperLu.from_key(key_4d_new)
    sender = Protocol(master)
    sender.skip_first_n(key_4d_new.n)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            cur_frame = cap.get(1) - 1
            frame_width, frame_height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if cur_frame % fps < quad_fps:
                msec_idx = int(cur_frame // fps * 1000)
            else:
                msec_idx = int(cur_frame // quad_fps * 250)
            if faceid in time_face_map[str(msec_idx)]:
                pos = video_face_map[faceid][str(msec_idx)]
                x1 = int(pos['x'])
                y1 = int(pos['y'])
                x2 = int(pos['x'] + pos['width'])
                y2 = int(pos['y'] + pos['height'])
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, int(frame_width)), min(y2, int(frame_height))
                frame[y1:y2, x1:x2] = sender.encrypt(frame[y1:y2, x1:x2])
            video_writer.write(frame)
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    key_revoke('8f2ff1f5-9e45-4ee9-a2ca-d716a679e05f', "bbt_10s.mp4")
