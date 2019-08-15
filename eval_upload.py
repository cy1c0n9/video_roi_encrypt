import templateGenerator
import encrypt_video
from time import time
import matplotlib.pyplot as plt
import numpy as np

test_folder = './videos/'
test_filename = ['qcrftest_0.mp4', 'qcrftest_1.mp4', 'qcrftest_2.mp4', 'qcrftest_3.mp4', 'qcrftest_4.mp4',
                 'qcrftest_5.mp4', 'qcrftest_6.mp4', 'crftest_0.mp4', 'crftest_1.mp4', 'crftest_2.mp4',
                 'hdtest_0.mp4', 'hdtest_1.mp4', 'hdtest_2.mp4', 'hdtest_3.mp4', "bbt_10s.mp4"]

"""
    test blur (template generate)
"""
# x_axis = [i for i in range(len(test_filename) )]
blur_time = []
encrypt_time = []

for j in range(1):
    for i in range(len(test_filename)):
        start = time()
        templateGenerator.process(test_folder, test_filename[i])
        end = time()
        print("video %d: blur time: %f" % (i, end-start))
        blur_time.append(end-start)

    for i in range(len(test_filename)):
        start = time()
        encrypt_video.process(test_filename[i], test_folder)
        end = time()
        print("video %d: encrypt time: %f" % (i, end-start))
        encrypt_time.append(end-start)
    print("round %s" % j)


x_axis = [i for i in range(len(blur_time))]
plt.plot(x_axis, blur_time, alpha=0.8, label='$blur$')
plt.plot(x_axis, encrypt_time, alpha=0.8, label='$encrypt$')

plt.xlabel('video idx')
plt.ylabel('time to deliver_key/sec')
plt.show()

# start = time()
# encrypt_video.deliver_key(test_folder, test_filename[12])
# print(time()-start)

blur_average = np.mean(blur_time)
blur_max = max(blur_time)
blur_var = np.var(blur_time)
encrypt_average = np.mean(encrypt_time)
encrypt_max = max(encrypt_time)
encrypt_var = np.var(encrypt_time)
print(' ')
print("time used:")
print("blur    : avg %s s, max %s s, variance %s" % (blur_average, blur_max, blur_var))
print("encrypt : avg %s s, max %s s, variance %s" % (encrypt_average, encrypt_max, encrypt_var))

"""
time used: 15 * 50 test case
blur    : avg 0.20512522697448732 s, max 0.5636217594146729 s, variance 0.026399538816050817
encrypt : avg 1.6287674779891967 s, max 3.8425214290618896 s, variance 1.3854661611484838
"""
