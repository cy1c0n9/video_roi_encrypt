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
        if len(key_l) != 6:
            print("Error: invalid key string!")
            return cls([0, 0, 0, 0, 0], 0)
        key_l[-1] = int(key_l[-1])
        return cls(key_l[0:-1], key_l[-1])

    @classmethod
    def from_list(cls, l: list):
        if len(l) != 6:
            print("Error: invalid key string!")
            return cls([0, 0, 0, 0, 0], 0)
        l[-1] = int(l[-1])
        return cls(l[0:-1], l[-1])

    @classmethod
    def rand_init(cls):
        x = np.random.uniform(-20, 20)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(0, 40)
        u = np.random.uniform(-100, 100)
        n = randint(1000, 3000)
        d = np.random.uniform(0, 1)
        return cls([x, y, z, u, d], n)

    def __str__(self):
        return "%s %s %s %s %s %d" % (self.xyz[0], self.xyz[1], self.xyz[2], self.xyz[3], self.xyz[4], self.n)


def generate_new_key(face_id) -> Key4D:
    import redis

    redis_host = "localhost"
    redis_port = 6379
    redis_password = ""
    r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
    new_key = Key4D.rand_init()
    r.hset("face_key_management", face_id, str(new_key))

    return new_key


def get_key_from_redis(face_id) -> Key4D:
    import redis

    redis_host = "localhost"
    redis_port = 6379
    redis_password = ""
    r = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
    key_s = r.hget("face_key_management", face_id)
    if key_s is not None:
        key = Key4D.from_str(key_s)
        return key
    else:
        raise AssertionError

# k = generate_new_key('1507bcbd-4b8f-422f-b657-c73d7e22a26b')
# print(k)
#
# k_p = get_key_from_redis('1')
# print(k_p)
