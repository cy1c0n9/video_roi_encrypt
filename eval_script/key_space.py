import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

x = np.linspace(1, 40, 40)
x_prime = np.linspace(1, 10, 10)

bf_pixel_permute_once = x ** (2*x)
bf_pixel_permute_twice = x ** (3*x)
bf_pixel_reorder = factorial(x_prime*x_prime, exact=False)

bf_key_permute_once = x * 0 + (10 ** 15) ** 4 * 10 ** 5
bf_key_permute_twice = x * 0 + (10 ** 15) ** 5 * 10 ** 5

standards_ads = x * 0 + 2 ** 256

fig, ax = plt.subplots()

# Using set_dashes() to modify dashing of an existing line
plt.yscale('log')
line0, = ax.plot(x, standards_ads, label='standard aes')
# line1, = ax.plot(x, bf_pixel_permute_once, label='brute force on permute once')
line2, = ax.plot(x, bf_pixel_permute_twice, label='brute force on 3 time permute')
line3, = ax.plot(x_prime, bf_pixel_reorder, label='brute force on reorder')
# line4, = ax.plot(x, bf_key_permute_once, label='brute force on 3D Lorenz key')
line5, = ax.plot(x, bf_key_permute_twice, label='brute force on 4D Lu key')

line0.set_dashes([5, 2])  # 5pt line, 2pt break

ax.legend()

plt.show()
