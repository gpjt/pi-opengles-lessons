#
# Copyright (c) 2017 Giles Thomas
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted.
#

import math
import numpy as np

def perspective(fovy, aspect, n, f):
    s = 1.0 / math.tan(math.radians(fovy) / 2.0)
    sx, sy = s / aspect, s
    zz = (f+n) / (n-f)
    zw = 2 * f * n / (n-f)
    return np.matrix([
        [sx,  0,  0,  0],
        [0,  sy,  0,  0],
        [0,   0, zz, zw],
        [0,   0, -1,  0]
    ])


def translate(xyz):
    x, y, z = xyz
    return np.matrix([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])
