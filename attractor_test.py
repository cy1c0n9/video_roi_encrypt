from attractor import Lorenz
from key import Key3D as Key
from attractor import Protocol
from attractor import ENCRYPT, DECRYPT
import numpy as np
import cv2
from time import time
import matplotlib.pyplot as plt

image_folder = "./images/"
filename = "loki.jpg"

# key = np.random.uniform(-10, 10), np.random.uniform(-15, 15), np.random.uniform(0, 40)
key_l = [11.381133082873731, 12.990637231862848, 8.822348261655519, 1000]

# init encoder
key_m = Key.from_list(key_l)
master = Lorenz.from_key(key_m)
sender = Protocol(master)
sender.skip_first_n(key_m.n)

# save key
print(str(key_m))
key_str = str(key_m)

# init decoder
key_s = Key.from_str(key_str)
slave = Lorenz.from_key(key_s)
receiver = Protocol(slave)
receiver.skip_first_n(key_s.n)
# print(master)
# print(slave)
print(' ')
# print(key_l[0])
# print(key_l[0] * 10000)

img1 = cv2.imread(image_folder+filename, cv2.IMREAD_UNCHANGED)
img2 = cv2.imread(image_folder+filename, cv2.IMREAD_UNCHANGED)
img3 = cv2.imread(image_folder+filename, cv2.IMREAD_UNCHANGED)

block_size = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
res_reorder = []
res_permute = []


# for bs in block_size:
#     # encrypt with reorder
#     start_time = time()
#     sender.reorder(img1[70:70+bs, 70:70+bs], ENCRYPT)
#     # img1[70:200, 40:160] = sender.encrypt(img1[70:200, 40:160])
#     t1 = time() - start_time
#     # print("--- Encrypt with reorder %s seconds ---" % t1)
#
#     # cv2.imshow('encrypt_reorder', img1)
#     # cv2.imwrite(image_folder+"permute_lorenz_encrypted.jpg", img1)
#
#     # encrypt with permute
#     start_time = time()
#     sender.encrypt(img1[70:70+bs, 70:70+bs])
#     # img3[70:200, 40:160] = sender.encrypt(img3[70:200, 40:160])
#     t2 = time() - start_time
#     # print("--- Encrypt with permute %s seconds ---" % t2)
#     # print(" ")
#     res_reorder.append(t1)
#     res_permute.append(t2)
#
#
# plt.plot(block_size, res_reorder, color="g", linestyle="--", marker="d", linewidth=1.0)
# plt.plot(block_size, res_permute, color="b", linestyle="--", marker="o", linewidth=1.0)
#
# plt.show()

# encrypt with reorder
start_time = time()
# sender.reorder(img1[70:70+bs, 70:70+bs], ENCRYPT)
# img2[70:200, 40:160] = sender.reorder(img1[70:200, 40:160], ENCRYPT)
t1 = time() - start_time
print("--- Encrypt with reorder %s seconds ---" % t1)

cv2.imshow('encrypt_reorder', img2)

# encrypt with permute
start_time = time()
# sender.encrypt(img1[70:70+bs, 70:70+bs])
img3[70:200, 40:160] = sender.encrypt(img3[70:200, 40:160])
t2 = time() - start_time
print("--- Encrypt with permute %s seconds ---" % t2)
print(" ")
cv2.imshow('encrypt_permute', img3)
cv2.imwrite(image_folder+"permute_lorenz_encrypted.jpg", img3)

# decrypt
img2 = cv2.imread(image_folder + "permute_lorenz_encrypted.jpg", cv2.IMREAD_UNCHANGED)
start_time = time()
# img2[70:200, 40:160] = receiver.reorder(img2[70:200, 40:160], DECRYPT)
img2[70:200, 40:160] = receiver.decrypt(img2[70:200, 40:160])
print("--- decrypt %s seconds ---" % (time() - start_time))
cv2.imshow('decrypt', img2)

cv2.waitKey()
cv2.destroyAllWindows()
