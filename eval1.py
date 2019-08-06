from attractor import Lorenz
from attractor import HyperLu
from key import Key4D
from key import Key3D
from attractor import Protocol
from attractor import ENCRYPT, DECRYPT
import numpy as np
import cv2
from time import time
import matplotlib.pyplot as plt
from eval_script.correlation_coefficients import cal_correlation_coefficients
from eval_script.correlation_coefficients import draw_security_evaluation

key_3d_list = [11.381133082873731, 12.990637231862848, 8.822348261655519, 1000]
key_4d_list = [11.381133082873731, 12.990637231862848, 8.822348261655519, 12.990637231862848, 0.58, 1000]

LOAD_BLUR_ENCRYPT_SAVE = True
LOAD_DECRYPT_SERVE = True
ORIGIN = True
BLUR = True
PERMUTE_ONCE = True
PERMUTE_TWICE = True
REORDER = True

if LOAD_BLUR_ENCRYPT_SAVE:
    """
        read original images
    """
    images = []
    # read eval1:
    image_folder = "./eval1/"
    file_suffix = "plain.jpg"
    for i in range(1, 39):
        img = cv2.imread(image_folder + str(i) + file_suffix, cv2.IMREAD_COLOR)
        # img = cv2.imread(image_folder + str(i) + file_suffix, cv2.IMREAD_GRAYSCALE)
        images.append(img)

    image_folder = "./eval2/"
    file_suffix = "plain_gray.png"
    for i in range(1, 81):
        img = cv2.imread(image_folder + str(i) + file_suffix, cv2.IMREAD_COLOR)
        # img = cv2.imread(image_folder + str(i) + file_suffix, cv2.IMREAD_GRAYSCALE)
        images.append(img)

    """
        prepare-phase: upload time
    """
    # blur layer
    if BLUR:
        image_folder = "./eval1_tmp/blur/"
        file_suffix = "blur_layer.png"
        for i in range(len(images)):
            blur_layer = images[i].copy()
            for _ in range(20):
                blur_layer = cv2.GaussianBlur(blur_layer, (11, 11), 0)
            cv2.imwrite(image_folder+str(i)+file_suffix, blur_layer)

    # permute once
    if PERMUTE_ONCE:
        image_folder = "./eval1_tmp/permute1/"
        file_suffix = "permute1.png"
        key_3d = Key3D.from_list(key_3d_list)
        master = Lorenz.from_key(key_3d)
        sender = Protocol(master)
        sender.skip_first_n(key_3d.n)
        for i in range(len(images)):
            print("permute once image %d" % i)
            img = sender.encrypt_permute_1(images[i].copy())
            cv2.imwrite(image_folder+str(i)+file_suffix, img)

    # permute twice
    if PERMUTE_TWICE:
        image_folder = "./eval1_tmp/permute2/"
        file_suffix = "permute2.png"
        key_4d = Key4D.from_list(key_4d_list)
        master = HyperLu.from_key(key_4d)
        sender = Protocol(master)
        sender.skip_first_n(key_4d.n)
        for i in range(len(images)):
            print("permute twice image %d" % i)
            img = sender.encrypt(images[i].copy())
            cv2.imwrite(image_folder+str(i)+file_suffix, img)
    # cv2.imshow('tmp', images[5])
    # cv2.waitKey()

    # reorder
    if REORDER:
        image_folder = "./eval1_tmp/reorder/"
        file_suffix = "reorder.png"
        key_4d = Key4D.from_list(key_4d_list)
        master = HyperLu.from_key(key_4d)
        sender = Protocol(master)
        sender.skip_first_n(key_4d.n)
        for i in range(len(images)):
            print("reorder image %d" % i)
            img = sender.reorder(images[i].copy(), ENCRYPT)
            cv2.imwrite(image_folder+str(i)+file_suffix, img)

if LOAD_DECRYPT_SERVE:
    size_threshold_lower = 0
    size_threshold = 400
    N = 10000
    """
        read original images
    """
    images = []
    # read eval1:
    image_folder = "./eval1/"
    file_suffix = "plain.jpg"
    times_origin = []
    cor_score_origin = []
    for i in range(1, 39):
        # start = time()
        img = cv2.imread(image_folder + str(i) + file_suffix, cv2.IMREAD_COLOR)
        if size_threshold_lower <= len(img[0]) < size_threshold:
            # times_origin.append(time() - start)
            times_origin.append(0)
            cor_score_origin.append(cal_correlation_coefficients(img, N))
        images.append(img)
    # read eval2:
    image_folder = "./eval2/"
    file_suffix = "plain_gray.png"
    for i in range(1, 81):
        # start = time()
        img = cv2.imread(image_folder + str(i) + file_suffix, cv2.IMREAD_COLOR)
        if len(img[0]) < size_threshold:
            # times_origin.append(time() - start)
            times_origin.append(0)
            cor_score_origin.append(cal_correlation_coefficients(img, N))
        images.append(img)
    # security for original image
    if ORIGIN:
        draw_security_evaluation(times_origin, cor_score_origin, 'firebrick')

    """
        serve-phase: request time
    """
    # blur layer
    if BLUR:
        image_folder = "./eval1_tmp/blur/"
        file_suffix = "blur_layer.png"
        image_folder_res = "./eval1_tmp/blur_res/"
        file_suffix_res = "blur.png"
        times_blur = []
        cor_score_blur = []
        for i in range(len(images)):
            print("blur image %d" % i)
            origin_img = images[i].copy()
            if size_threshold_lower <= len(origin_img[0]) < size_threshold:
                start = time()
                template_layer = cv2.imread(image_folder + str(i) + file_suffix, cv2.IMREAD_COLOR)
                origin_img[:, :] = template_layer
                end = time()

                times_blur.append(end - start)
                cor_score_blur.append(cal_correlation_coefficients(origin_img, N))
                cv2.imwrite(image_folder_res + str(i) + file_suffix_res, origin_img)
        draw_security_evaluation(times_blur, cor_score_origin, 'slategray', 'blur layer')

    # permute once
    if PERMUTE_ONCE:
        image_folder = "./eval1_tmp/permute1/"
        file_suffix = "permute1.png"
        image_folder_res = "./eval1_tmp/permute1_decrypt/"
        file_suffix_res = "permute1_decrypt.png"
        key_3d = Key3D.from_list(key_3d_list)
        slave = Lorenz.from_key(key_3d)
        receiver = Protocol(slave)
        receiver.skip_first_n(key_3d.n)
        times_permute_once = []
        cor_score_permute_once = []
        for i in range(len(images)):
            print("permute once image %d" % i)
            img = cv2.imread(image_folder + str(i) + file_suffix, cv2.IMREAD_COLOR)
            if size_threshold_lower <= len(img[0]) < size_threshold:
                cor = cal_correlation_coefficients(img, N)

            start = time()
            img = receiver.decrypt_permute_1(img)
            end = time()

            if size_threshold_lower <= len(img[0]) < size_threshold:
                times_permute_once.append(end - start)
                cor_score_permute_once.append(cor)
                cv2.imwrite(image_folder_res + str(i) + file_suffix_res, img)
        draw_security_evaluation(times_permute_once, cor_score_permute_once, 'g', 'permute once')

    # permute twice
    if PERMUTE_TWICE:
        image_folder = "./eval1_tmp/permute2/"
        file_suffix = "permute2.png"
        image_folder_res = "./eval1_tmp/permute2_decrypt/"
        file_suffix_res = "permute2_decrypt.png"
        key_4d = Key4D.from_list(key_4d_list)
        slave = HyperLu.from_key(key_4d)
        receiver = Protocol(slave)
        receiver.skip_first_n(key_4d.n)
        times_permute_twice = []
        cor_score_permute_twice = []
        for i in range(len(images)):
            print("permute twice image %d" % i)
            img = cv2.imread(image_folder + str(i) + file_suffix, cv2.IMREAD_COLOR)
            if size_threshold_lower <= len(img[0]) < size_threshold:
                cor = cal_correlation_coefficients(img, N)

            start = time()
            img = receiver.decrypt(img)
            end = time()

            if size_threshold_lower <= len(img[0]) < size_threshold:
                times_permute_twice.append(end - start)
                cor_score_permute_twice.append(cor)
                cv2.imwrite(image_folder_res + str(i) + file_suffix_res, img)
        draw_security_evaluation(times_permute_twice, cor_score_permute_twice, 'mediumblue', 'permute twice')

    # reorder
    if REORDER:
        image_folder = "./eval1_tmp/reorder/"
        file_suffix = "reorder.png"
        image_folder_res = "./eval1_tmp/reorder_decrypt/"
        file_suffix_res = "reorder_decrypt.png"
        key_4d = Key4D.from_list(key_4d_list)
        slave = HyperLu.from_key(key_4d)
        receiver = Protocol(slave)
        receiver.skip_first_n(key_4d.n)
        times_reorder = []
        cor_score_reorder = []
        for i in range(len(images)):
            print("reorder image %d" % i)
            img = cv2.imread(image_folder + str(i) + file_suffix, cv2.IMREAD_COLOR)
            if size_threshold_lower <= len(img[0]) < size_threshold:
                cor = cal_correlation_coefficients(img, N)

            start = time()
            img = receiver.reorder(img, DECRYPT)
            end = time()

            if size_threshold_lower <= len(img[0]) < size_threshold:
                times_reorder.append(end - start)
                cor_score_reorder.append(cor)
                cv2.imwrite(image_folder_res + str(i) + file_suffix_res, img)
        draw_security_evaluation(times_reorder, cor_score_reorder, 'purple', 're-order')


plt.xlabel('sec/face')
plt.ylabel('correlation coefficients')
plt.show()
