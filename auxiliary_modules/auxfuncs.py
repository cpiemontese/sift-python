#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from math import sqrt
from json import loads
from math import pi, sqrt, exp


# ------------------------------------------------------------------------------------------- #
#                                  Error Auxiliary Functions                                  #
# ------------------------------------------------------------------------------------------- #


def exitWithErr(errType, errMsg):
    print '{0}: {1}\nExiting now.'.format(errType, errMsg)
    exit()


# -------------------------------------------------------------------------------------------- #
#                               Mathematical Auxiliary Functions                               #
# -------------------------------------------------------------------------------------------- #


def normalize(vec, isMatrix = False):
    if isMatrix:
        normMat = []
        for row in vec:
            normMat.append(normalize(row))
        return normMat

    squaresSum = 0
    for el in vec:
        squaresSum += el**2
    s = sqrt(squaresSum)
    if s == .0:
        return vec  # array is all zeros
    weight = 1./s
    return map(lambda x: x*weight, vec)
