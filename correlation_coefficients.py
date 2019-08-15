import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import random
from math import sqrt


def cal_correlation_coefficients(image, n):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height = len(gray)
    width = len(gray[0])
    print('calculate correlation coefficients for size [%d, %d]' % (height, width))
    direction = [(-1, -1), (-1,  0), (-1,  1),
                 ( 0, -1), ( 0,  1),
                 ( 1, -1), ( 1,  0), ( 1,  0)]
    x = []
    y = []
    for _ in range(n):
        xi, xj, d = random.randint(1, height-1), random.randint(1, width-1), random.randint(8)
        yi, yj = xi + direction[d][0], xj + direction[d][1]
        x.append(gray[xi][xj])
        y.append(gray[yi][yj])
    E_x = sum(x) / n
    E_y = sum(y) / n
    # print(x)
    # print(y)

    cov = 0
    D_x = 0
    D_y = 0
    for i in range(n):
        cov += (x[i] - E_x) * (y[i] - E_y)
        D_x += (x[i] - E_x) * (x[i] - E_x)
        D_y += (y[i] - E_y) * (y[i] - E_y)
    r = cov / sqrt(D_x * D_y)
    return 1.0 - r


def draw_correlation_for_img(image, colour: str):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height = len(gray)
    width = len(gray[0])
    print(height, width)
    x = []
    y = []
    for i in range(height-1):
        for j in range(width-1):
            x.append(gray[i][j])
            y.append(gray[i+1][j+1])
    plt.scatter(x, y, alpha=0.6, s=1, c=colour)
    # plt.show()


def draw_security_evaluation(times, cor_score, colour, label='$original$'):
    plt.scatter(times, cor_score, alpha=0.4, s=25, c=colour, label=label)
    plt.legend()


# image_folder = "./images/"
# filename = "loki.jpg"
# img1 = cv2.imread(image_folder+filename, cv2.IMREAD_UNCHANGED)
# gray1 = cv2.cvtColor(img1[70:200, 40:160], cv2.COLOR_BGR2GRAY)
# res1 = cal_correlation_coefficients(gray1, 10)
# draw_correlation_for_img(gray1, 'r')
# filename = "permute_lorenz_encrypted.jpg"
# img2 = cv2.imread(image_folder+filename, cv2.IMREAD_UNCHANGED)
# gray2 = cv2.cvtColor(img2[70:200, 40:160], cv2.COLOR_BGR2GRAY)
# res2 = cal_correlation_coefficients(gray2, 10)
# draw_correlation_for_img(gray2, 'b')
# # print()
# # print(res)
# plt.show()
