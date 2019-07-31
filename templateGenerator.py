# !/usr/bin/env python3

import redis
import json
import cv2

redis_host = "localhost"
redis_port = 6379
redis_password = ""
video_folder = "./videos/"
filename = "103792584148574_1562618673.mp4"


def process():
    """
    pre process data
    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
    0                         500                                 1000
    :return:
    """
    try:
        # The decode_responses flag here directs the client to convert the responses from Redis into Python strings
        r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)

        video_face_map = json.loads(r.hget("video_face_map", filename))
        time_face_map = json.loads(r.hget('time_face_map', filename))
        cap = cv2.VideoCapture(video_folder + filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        quad_fps = int((fps + 3) / 4)
        fps = int(fps + 0.5)
        print(fps)
        print(quad_fps)
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
        # Read until video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # stamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                msec_idx = -1
                cur_frame = cap.get(1) - 1
                frame_width, frame_height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                # print(cur_frame)
                if cur_frame % fps == 0:
                    msec_idx = int(cur_frame // fps * 1000)
                elif cur_frame % quad_fps == 0:
                    msec_idx = int(cur_frame // quad_fps * 250)
                else:
                    pass
                print(msec_idx)

                if msec_idx != -1:
                    for fid, idx in time_face_map[str(msec_idx)].items():
                        pos = video_face_map[fid][str(msec_idx)]
                        """
                        $height = floatval($height) * 1.2;
                        $width = floatval($width) * 1.2;
                        $x1 = floatval($x) - 1 / 12 * $width;
                        $y1 = floatval($y) - 1 / 12 * $height;
                
                        $x1 = $x1 > 0 ? $x1 : 0;
                        $y1 = $y1 > 0 ? $y1 : 0;
                        $x1 = $x1 < $src_w ? $x1 : $src_w-1;
                        $y1 = $y1 < $src_h ? $y1 : $src_h-1;
                        $width = $x1 + $width < $src_w ? $width : $src_w-1-$x1;
                        $height = $y1 + $height < $src_h ? $height : $src_h-1-$y1;
                        """
                        x1 = int(pos['x'] - 1 / 12 * pos['width'])
                        y1 = int(pos['y'] - 1 / 12 * pos['height'])
                        x2 = int(pos['x'] + 1.2 * pos['width'])
                        y2 = int(pos['y'] + 1.2 * pos['height'])
                        x1, y1 = max(x1, 0), max(y1, 0)
                        x2, y2 = min(x2, frame_width), min(y2, frame_height)
                        crop_img = frame[y1:y2, x1:x2]
                        for _ in range(20):
                            crop_img = cv2.GaussianBlur(crop_img, (11, 11), 0)
                        # cv2.imshow('corp', crop_img)
                        cv2.imwrite('./video_faces/' + fid + '_' + str(msec_idx) + '.png', crop_img)
                # Display the resulting frame
                # cv2.imshow('Frame', frame)
                # timestamps.append(stamp)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break

        cap.release()

        # for i, ts in enumerate(timestamps):
        #     print('Frame %d :' % i, ts)
        # When everything done, release the video capture object

        # Closes all the frames
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)


if __name__ == '__main__':
    process()
