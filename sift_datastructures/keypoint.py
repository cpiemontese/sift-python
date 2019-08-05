#!/usr/bin/python
# -*- coding: utf-8 -*-

class Keypoint:
    def __init__(self, x = -1., y = -1., s = -1.):
        self.x = x
        self.y = y
        self.scale = s
        self.orientation = None
        self.octave = None

    def __str__(self):
        return "({0}, {1}, {2}), orientation: {3}, octave: {4}".format(self.x, self.y, self.scale, self.orientation, self.octave)

    def addOffset(self, offset):
        x, y, s = offset
        self.x += x[0]
        self.y += y[0]
        self.scale += s[0]

    def getCoords(self):
        return [self.x, self.y, self.scale]

    def getRoundedCoords(self):
        return [int(round(self.x)), int(round(self.y)), int(round(self.scale))]

    def copy(self):
        copy = Keypoint(self.x, self.y, self.scale)
        copy.octave = self.octave
        copy.orientation = self.orientation
        return copy
