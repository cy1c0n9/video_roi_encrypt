from abc import ABCMeta
from abc import abstractmethod
from key import Key3D
from key import Key4D
from random import randint
from time import time
import cv2
import numpy as np

ENCRYPT = 1
DECRYPT = -1


class Attractor:
    __metaclass__ = ABCMeta

    def __init__(self, xyz: list):
        self.xyz = xyz

    @classmethod
    @abstractmethod
    def from_list(cls, l: list): pass

    @classmethod
    @abstractmethod
    def from_key(cls, key: Key3D): pass

    @abstractmethod
    def __str__(self): pass

    @abstractmethod
    def get_vec(self, vec, t) -> np.ndarray: pass

    @abstractmethod
    def check_valid(self) -> bool: pass


class Lorenz(Attractor):
    """
    Lorenz attractor, the differential equation does not change according to t
    x domain: -10, 10
    y domain: -15, 15
    z domain:   0, 40
    """

    def __init__(self, xyz):
        Attractor.__init__(self, xyz)
        self.a = 10
        self.b = 28
        self.c = 8 / 3

    @classmethod
    def from_list(cls, l: list):
        if len(l) != 3:
            print("Warning: invalid input!, initialize to zero")
            return cls([0, 0, 0])
        return cls(l)

    @classmethod
    def from_key(cls, key: Key3D):
        return cls.from_list(key.xyz)

    def get_vec(self, vec: np.ndarray, t) -> np.ndarray:
        """
        get dx/dt, dy/dt, dz/dt
        :return: np.array(dx/dt, dy/dt, dz/dt)
        """
        x = vec[0]
        y = vec[1]
        z = vec[2]
        xr = self.a * (y - x)
        yr = x * (self.b - z) - y
        zr = x * y - self.c * z
        return np.array([xr, yr, zr])

    def __str__(self):
        return "%s %s %s" % (self.xyz[0], self.xyz[1], self.xyz[2])

    def check_valid(self) -> bool:
        return self.xyz and len(self.xyz) == 3


class HyperLu(Attractor):
    """
    Lorenz attractor, the differential equation does not change according to t
    x domain: -20, 20
    y domain: -20, 20
    z domain:   0, 40
    u domain:-100, 100
    the step size used in Hyper Lu attractor should  <= 0.02
    """

    def __init__(self, xyz):
        Attractor.__init__(self, xyz)
        self.a = 36.0
        self.b = 3.0
        self.c = 20.0
        self.d = 0.5

    @classmethod
    def from_list(cls, l: list):
        if len(l) != 4:
            print("Warning: invalid input!, initialize to zero")
            return cls([0, 0, 0, 0])
        return cls(l)

    @classmethod
    def from_key(cls, key: Key4D):
        return cls.from_list(key.xyz)

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

    def __str__(self):
        return "%s %s %s %s" % (self.xyz[0], self.xyz[1], self.xyz[2], self.xyz[3])

    def check_valid(self) -> bool:
        return self.xyz and len(self.xyz) == 4


class RungeKutta4:

    def __init__(self, attractor):
        self.attractor = attractor

    # def solve(self, x, y, z, t, h):
    #     """
    #     solve the differential equations
    #
    #     :param x: x(t)
    #     :param y: y(t)
    #     :param z: z(t)
    #     :param t: current t
    #     :param h: step height
    #     :return: x(t+h)-x(t), y(t+h)-y(t), z(t+h)-z(t)
    #     """
    #     k1 = h * self.attractor.get_x(x, y, z, t)
    #     l1 = h * self.attractor.get_y(x, y, z, t)
    #     m1 = h * self.attractor.get_z(x, y, z, t)
    #     k2 = h * self.attractor.get_x(x + k1 / 2, y + l1 / 2,
    #                                   z + m1 / 2, t + h / 2)
    #     l2 = h * self.attractor.get_y(x + k1 / 2, y + l1 / 2,
    #                                   z + m1 / 2, t + h / 2)
    #     m2 = h * self.attractor.get_z(x + k1 / 2, y + l1 / 2,
    #                                   z + m1 / 2, t + h / 2)
    #     k3 = h * self.attractor.get_x(x + k2 / 2, y + l2 / 2,
    #                                   z + m2 / 2, t + h / 2)
    #     l3 = h * self.attractor.get_y(x + k2 / 2, y + l2 / 2,
    #                                   z + m2 / 2, t + h / 2)
    #     m3 = h * self.attractor.get_z(x + k2 / 2, y + l2 / 2,
    #                                   z + m2 / 2, t + h / 2)
    #     k4 = h * self.attractor.get_x(x + k3, y + l3, z + m3, t + h)
    #     l4 = h * self.attractor.get_y(x + k3, y + l3, z + m3, t + h)
    #     m4 = h * self.attractor.get_z(x + k3, y + l3, z + m3, t + h)
    #     xr = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    #     yr = (l1 + 2 * l2 + 2 * l3 + l4) / 6
    #     zr = (m1 + 2 * m2 + 2 * m3 + m4) / 6
    #     return np.array([xr, yr, zr])

    def solve_vector(self, vec, t, h):
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


class Protocol:

    def __init__(self, attractor: Attractor):
        self.attractor = attractor
        self.rk4 = RungeKutta4(attractor)

    def skip_first_n(self, n: int, h=0.01) -> None:
        """
        skip first n iteration to obtain a chaos orbit
        :param n: number of the iteration to skip
        :param h: step height
        :return: None
        """
        start_time = time()
        if self.attractor.check_valid():
            xyz = self.attractor.xyz
        else:
            raise ValueError
        sequence_i = np.array(xyz)
        i = 0
        t = 0.0
        while i < n:
            sequence_i = sequence_i + self.rk4.solve_vector(sequence_i, t, h)
            t += h
            i += 1
        self.attractor.xyz = sequence_i.tolist()
        print("--- skip first n in %s seconds ---" % (time() - start_time))
        return None

    def get_sequence(self, n: int, h=0.01) -> np.ndarray:
        """
        generate n element
        :param n: n element to be generated
        :param h: step height
        :return:
        """
        # start_time = time()
        sequence = []
        if self.attractor.check_valid():
            xyz = self.attractor.xyz
        else:
            raise ValueError
        sequence_i = np.array(xyz)
        i = 0
        t = 0.0
        while i < n:
            # x, y, z = sequence_i
            # sequence_i = sequence_i + self.rk4.solve(x, y, z, t, h)
            sequence_i = sequence_i + self.rk4.solve_vector(sequence_i, t, h)
            sequence.append(sequence_i)
            t += h
            i += 1
        self.attractor.xyz = sequence_i.tolist()
        # print("--- get sequence in %s seconds ---" % (time() - start_time))
        return np.array(sequence)

    def permute(self, matrix, sequence_height, sequence_width, code):
        # start_time = time()
        if code == 1:
            # self.make_roll(b.T, g.T, r.T, sequence_width, code)
            # self.make_roll(b, g, r, sequence_height, code)
            a = np.transpose(matrix, axes=[1, 0, 2])
            self.make_roll_rgb(a, sequence_width, code, 0)
            self.make_roll_rgb(matrix, sequence_height, code, 0)
            # print("--- permute in %s seconds ---" % (time() - start_time))
            return matrix
        elif code == -1:
            self.make_roll_rgb(matrix, sequence_height, code, 0)
            a = np.transpose(matrix, axes=[1, 0, 2])
            self.make_roll_rgb(a, sequence_width, code, 0)
            # self.make_roll(b, g, r, sequence_height, code)
            # self.make_roll(b.T, g.T, r.T, sequence_width, code)
            # print("--- permute in %s seconds ---" % (time() - start_time))
            return matrix
        else:
            raise AttributeError

    def permute_twice(self, matrix, sequence_height, sequence_width, code):
        if code == 1:
            a = np.transpose(matrix, axes=[1, 0, 2])
            self.make_roll_rgb(a, sequence_width, code, 0)
            self.make_roll_rgb(matrix, sequence_height, code, 0)
            # a = np.transpose(matrix, axes=[1, 0, 2])
            self.make_roll_rgb(a, sequence_width, code, 2)
            self.make_roll_rgb(matrix, sequence_height, code, 2)
        elif code == -1:
            self.make_roll_rgb(matrix, sequence_height, code, 2)
            a = np.transpose(matrix, axes=[1, 0, 2])
            self.make_roll_rgb(a, sequence_width, code, 2)
            self.make_roll_rgb(matrix, sequence_height, code, 0)
            # a = np.transpose(matrix, axes=[1, 0, 2])
            self.make_roll_rgb(a, sequence_width, code, 0)
        else:
            raise AttributeError
        return matrix

    @staticmethod
    def make_roll(b, g, r, sequence, direction):
        length = len(sequence)
        for i in range(length):
            b[i] = np.roll(b[i], int(sequence[i][1]) * direction, axis=0)
            g[i] = np.roll(g[i], int(sequence[i][1]) * direction, axis=0)
            r[i] = np.roll(r[i], int(sequence[i][1]) * direction, axis=0)

    @staticmethod
    def make_roll_rgb(img, sequence, direction, coordinate):
        # start_time = time()
        length = len(sequence)
        for i in range(length):
            img[i] = np.roll(img[i], int(sequence[i][coordinate]) * direction, axis=0)
        # print("--- roll in %s seconds ---" % (time() - start_time))

    def block_operation(self, img_block, height: int, width: int, code, h=0.1):
        if height != len(img_block) or width != len(img_block[0]):
            raise ValueError
        sequence_h = np.rint(self.get_sequence(height, h) * 10000) % height
        sequence_w = np.rint(self.get_sequence(width, h) * 10000) % width
        img_block = self.permute(img_block, sequence_h, sequence_w, code)
        return img_block

    def encrypt(self, img, h=0.01):
        height = len(img)
        width = len(img[0])
        sequence_h = np.rint(self.get_sequence(height, h) * 10000) % height
        sequence_w = np.rint(self.get_sequence(width, h) * 10000) % width
        img = self.permute_twice(img, sequence_h, sequence_w, ENCRYPT)
        return img

    def decrypt(self, img, h=0.01):
        height = len(img)
        width = len(img[0])
        sequence_h = np.rint(self.get_sequence(height, h) * 10000) % height
        sequence_w = np.rint(self.get_sequence(width, h) * 10000) % width
        img = self.permute_twice(img, sequence_h, sequence_w, DECRYPT)
        return img

    """
    def encrypt_block(self, img, block_size=32):
        height = len(img)
        width = len(img[0])
        block_y = height // block_size + 1 if height % block_size else height // block_size
        block_x = width // block_size + 1 if width % block_size else width // block_size
        for i in range(block_y):
            for j in range(block_x):
                y1 = block_size * i
                x1 = block_size * j
                y2 = block_size + y1 if i < block_y - 1 else height
                x2 = block_size + x1 if j < block_x - 1 else width
                img[y1:y2, x1:x2] = self.block_operation(img[y1:y2, x1:x2], y2-y1, x2-x1, ENCRYPT)
        # img = self.block_operation(img, height, width, ENCRYPT)
        return img

    def decrypt_block(self, img, block_size=32):
        height = len(img)
        width = len(img[0])
        block_y = height // block_size + 1 if height % block_size else height // block_size
        block_x = width // block_size + 1 if width % block_size else width // block_size
        for i in range(block_y):
            for j in range(block_x):
                y1 = block_size * i
                x1 = block_size * j
                y2 = block_size + y1 if i < block_y - 1 else height
                x2 = block_size + x1 if j < block_x - 1 else width
                img[y1:y2, x1:x2] = self.block_operation(img[y1:y2, x1:x2], y2 - y1, x2 - x1, DECRYPT)
        # img = self.block_operation(img, height, width, ENCRYPT)
        return img
    """

    def rand_perm_map(self, length, h=0.01) -> np.ndarray:
        # start_time = time()
        idx_mat = np.arange(length)
        sequence = np.rint(self.get_sequence(length, h) * 100000)

        for i in range(length-1, -1, -1):
            r = int(sequence[i][2] % (i+1))
            idx_mat[i], idx_mat[r] = idx_mat[r], idx_mat[i]
        # print("--- generate map in %s seconds ---" % (time() - start_time))
        return idx_mat

    def reorder(self, img, code=ENCRYPT):
        height = len(img)
        width = len(img[0])
        length = height * width
        idx_map = self.rand_perm_map(length)
        res = np.zeros(img.shape, np.uint8)

        # start_time = time()
        if code == ENCRYPT:
            for i in range(length):
                res[i // width][i % width] = img[idx_map[i] // width][idx_map[i] % width]
        elif code == DECRYPT:
            for i in range(length):
                res[idx_map[i] // width][idx_map[i] % width] = img[i // width][i % width]
        else:
            raise AttributeError
        # print("--- reorder in %s seconds ---" % (time() - start_time))
        return res
