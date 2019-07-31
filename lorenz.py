# -*- coding: utf-8 -*-
from scipy.integrate import odeint
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from time import time


class Attractor:
    __metaclass__ = ABCMeta

    def __init__(self, x_0=None, y_0=None, z_0=None, u_0=None):
        self.x_0 = x_0
        self.y_0 = y_0
        self.z_0 = z_0
        self.u_0 = u_0

    def __str__(self):
        return "%s %s %s %s" % (self.x_0, self.y_0, self.z_0, self.u_0)

    @abstractmethod
    def get_vec(self, vec, t) -> np.ndarray: pass


class RungeKutta4:

    def __init__(self, attractor):
        self.attractor = attractor

    def solve(self, vec, t, h):
        """
        solve the differential equations

        :param vec: x/y/z (t)
        :param t: current t
        :param h: step height
        :return: x(t+h)-x(t), y(t+h)-y(t), z(t+h)-z(t)
        """
        k1 = h * self.attractor.get_vec(vec, t)
        k2 = h * self.attractor.get_vec(vec + k1 / 2, t + h / 2)
        k3 = h * self.attractor.get_vec(vec + k2 / 2, t + h / 2)
        k4 = h * self.attractor.get_vec(vec + k3, t + h)
        return (k1 + 2 * k2 + 2 * k3 + k4) / 6


class HyperLu(Attractor):
    """
    Lorenz attractor, the differential equation does not change according to t
    x domain: -10, 10
    y domain: -15, 15
    z domain:   0, 40
    """
    def __init__(self, x_0=None, y_0=None, z_0=None, u_0=None):
        Attractor.__init__(self, x_0, y_0, z_0, u_0)
        self.a = 36.0
        self.b = 3.0
        self.c = 20.0
        self.d = 0.5

    @classmethod
    def from_list(cls, l: list):
        if len(l) < 4:
            print("Warning: invalid input!, initialize randomly")
            return cls()
        return cls(l[0], l[1], l[2], l[3])

    def get_vec(self, vec: np.ndarray, t) -> np.ndarray:
        x = vec[0]
        y = vec[1]
        z = vec[2]
        u = vec[3]
        xr = self.a * (y - x) + u
        yr = - x * z + self.c * y
        zr = x * y - self.b * z
        ur = x * z + self.d * u
        return np.array([xr, yr, zr, ur])


class HyperLorenz(Attractor):
    """
    Lorenz attractor, the differential equation does not change according to t
    x domain: -10, 10
    y domain: -15, 15
    z domain:   0, 40
    """
    def __init__(self, x_0=None, y_0=None, z_0=None, u_0=None):
        Attractor.__init__(self, x_0, y_0, z_0, u_0)
        self.a = 10.0
        self.b = 8.0 / 3.0
        self.c = 28.0
        self.d = -1.15

    @classmethod
    def from_list(cls, l: list):
        if len(l) < 4:
            print("Warning: invalid input!, initialize randomly")
            return cls()
        return cls(l[0], l[1], l[2], l[3])

    def get_vec(self, vec: np.ndarray, t) -> np.ndarray:
        x = vec[0]
        y = vec[1]
        z = vec[2]
        u = vec[3]
        xr = self.a * (y - x) + u
        yr = self.c * x - x * z - y
        zr = x * y - self.b * z
        ur = - y * z + self.d * u
        return np.array([xr, yr, zr, ur])


attractor = HyperLu.from_list([8.0, 4.00000, 4.0, 3.0])
rk4 = RungeKutta4(attractor)
start = time()
t = np.arange(0, 50.0, 0.01)

sequence = []
x = attractor.x_0
y = attractor.y_0
z = attractor.z_0
u = attractor.u_0

sequence_i = np.array([x, y, z, u])
for ti in t:
    sequence_i = sequence_i + rk4.solve(sequence_i, t, 0.01)
    sequence.append(sequence_i)
print("--- generate %s seconds ---" % (time() - start))

track1 = np.array(sequence)

attractor = HyperLu.from_list([8.0, 4.00000001, 4.0, 3.0])
rk4_2 = RungeKutta4(attractor)

sequence2 = []
x = attractor.x_0
y = attractor.y_0
z = attractor.z_0
u = attractor.u_0

sequence_i = np.array([x, y, z, u])
for ti in t:
    sequence_i = sequence_i + rk4_2.solve(sequence_i, t, 0.01)
    sequence2.append(sequence_i)

track2 = np.array(sequence2)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
# ax.plot(track1[3900:3999, 0], track1[3900:3999, 1], track1[3900:3999, 2])
# ax.plot(track2[3900:3999, 0], track2[3900:3999, 1], track2[3900:3999, 2])
ax.plot(track1[:, 0], track1[:, 1], track1[:, 3])
ax.plot(track2[:, 0], track2[:, 1], track2[:, 3])
plt.show()
