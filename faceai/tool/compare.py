#
# coding=utf-8
# description: calculate and compare the distance of multiple 2-d vectors.
#

import numpy as np


def distance(vector1, vector2):
    if not isinstance(vector1, np.ndarray):
        vector1 = np.array(vector1)
    if not isinstance(vector2, np.ndarray):
        vector2 = np.array(vector2)

    return np.linalg.norm(vector1 - vector2)


def compare(vectors, vector):
    distances = []
    for v in vectors:
        distances.append(distance(v, vector))
    return np.argsort(distances)


if __name__ == '__main__':
    pass
