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

key_3d_list = [11.381133082873731, 12.990637231862848, 8.822348261655519, 1000]
key_4d_list = [11.381133082873731, 12.990637231862848, 8.822348261655519, 12.990637231862848, 1000]

"""
    read all images
"""
images = []
# read eval1:
image_folder = "./eval1/"
file_suffix = "plain.jpg"
for i in range(1, 20):
    img = cv2.imread(image_folder + str(i) + file_suffix, cv2.IMREAD_UNCHANGED)
    # img = cv2.imread(image_folder + str(i) + file_suffix, cv2.IMREAD_GRAYSCALE)
    images.append(img)

image_folder = "./eval2/"
file_suffix = "plain_gray.png"
for i in range(1, 41):
    img = cv2.imread(image_folder + str(i) + file_suffix, cv2.IMREAD_UNCHANGED)
    # img = cv2.imread(image_folder + str(i) + file_suffix, cv2.IMREAD_GRAYSCALE)
    images.append(img)

"""
    prepare-phase: upload time
"""
# blur layer
image_folder = "./eval1_tmp/blur/"
file_suffix = "blur_layer.png"
for i in range(len(images)):
    blur_layer = images[i]
    for _ in range(20):
        blur_layer = cv2.GaussianBlur(blur_layer, (11, 11), 0)
    cv2.imwrite(image_folder+str(i)+file_suffix, blur_layer)

# permute once
image_folder = "./eval1_tmp/permute1/"
file_suffix = "permute1.png"
key_3d = Key3D.from_list(key_3d_list)
master = Lorenz.from_key(key_3d)
sender = Protocol(master)
sender.skip_first_n(key_3d.n)
for i in range(len(images)):
    img = sender.encrypt_permute_1(images[i])
    cv2.imwrite(image_folder+str(i)+file_suffix, img)


