from abc import ABCMeta
from abc import abstractmethod
from random import randint
import numpy as np


class Key:
    __metaclass__ = ABCMeta

    def __init__(self, xyz: list, n):
        self.xyz = xyz
        self.n = n

    @classmethod
    @abstractmethod
    def from_str(cls, s: str): pass

    @classmethod
    @abstractmethod
    def from_list(cls, l: list): pass

    @classmethod
    @abstractmethod
    def rand_init(cls): pass

    @abstractmethod
    def __str__(self): pass


class Key3D(Key):
    def __init__(self, xyz, n):
        Key.__init__(self, xyz, n)

    @classmethod
    def from_str(cls, s: str):
        key_l = list(map(float, s.split(' ')))
        if len(key_l) != 4:
            print("Error: invalid key string!")
            return cls([0, 0, 0], 0)
        key_l[3] = int(key_l[3])
        return cls(key_l[0:-1], key_l[-1])

    @classmethod
    def from_list(cls, l: list):
        if len(l) != 4:
            print("Error: invalid key string!")
            return cls([0, 0, 0], 0)
        l[3] = int(l[3])
        return cls(l[0:-1], l[-1])

    @classmethod
    def rand_init(cls):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-15, 15)
        z = np.random.uniform(0, 40)
        n = randint(1000, 3000)
        return cls([x, y, z], n)

    def __str__(self):
        return "%s %s %s %d" % (self.xyz[0], self.xyz[1], self.xyz[2], self.n)


class Key4D(Key):
    def __init__(self, xyz, n):
        Key.__init__(self, xyz, n)

    @classmethod
    def from_str(cls, s: str):
        key_l = list(map(float, s.split(' ')))
        if len(key_l) != 5:
            print("Error: invalid key string!")
            return cls([0, 0, 0, 0], 0)
        key_l[4] = int(key_l[4])
        return cls(key_l[0:-1], key_l[4])

    @classmethod
    def from_list(cls, l: list):
        if len(l) != 5:
            print("Error: invalid key string!")
            return cls([0, 0, 0, 0], 0)
        l[4] = int(l[4])
        return cls(l[0:-1], l[4])

    @classmethod
    def rand_init(cls):
        x = np.random.uniform(-20, 20)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(0, 40)
        u = np.random.uniform(-100, 100)
        n = randint(1000, 3000)
        return cls([x, y, z, u], n)

    def __str__(self):
        return "%s %s %s %s %d" % (self.xyz[0], self.xyz[1], self.xyz[2], self.xyz[3], self.n)

