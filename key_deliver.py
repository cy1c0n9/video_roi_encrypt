# !/usr/bin/env python3

import redis
import json

from key import get_key_from_redis
from time import time
import sys


def deliver_key(requester, filename):
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
        data_store = {}
        # The decode_responses flag here directs the client to convert the responses from Redis into Python strings
        r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)

        video_face_map = json.loads(r.hget("video_face_map", filename))
        time_face_map = json.loads(r.hget('time_face_map', filename))

        key_dict = {}
        video_face_trim = {}
        for fid in video_face_map.keys():
            face_policy = json.loads(r.hget('video_face_policy_map', fid))
            if requester in face_policy:
                key_4d = get_key_from_redis(fid)
                key_dict[fid] = str(key_4d)
                video_face_trim[fid] = video_face_map[fid]
        data_store['video_face_map'] = video_face_trim
        data_store['face_key'] = key_dict

        time_face_trim = {}
        for t, faces in time_face_map.items():
            res = []
            for fid in faces:
                if fid in key_dict:
                    # print(faces)
                    res.append(fid)
            time_face_trim[t] = res
        data_store['time_face_map'] = time_face_trim

        with open('./json/' + filename + '.json', 'w') as f:
            json.dump(data_store, f)

    except Exception as e:
        print(e)


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print('error: system argument')
    #     raise AttributeError
    # else:
    #     file_name = sys.argv[1]
    # deliver_key(filename=file_name)
    start = time()
    deliver_key('103792584148574', "bbt_10s.mp4")
    # deliver_key('108491877063719', "bbt_10s.mp4")
    print("time: %s s" % (time() - start))
