#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
This module implements SIFT features visualization and pyramid navigation
"""


import matplotlib.pyplot as plt

from matplotlib.cm import Greys_r
from math import cos, sin, radians

class SiftVisualizer:
    """
    SiftVisualizer class implements:
        - keypoint visualization on an image (showKeypoints)
        - pyramid navigation (navigatePyramid)
    """
    def __init__(self, pyrToVisualize = None, kpsToVisualize = None):
        # variables linked to detector data that needs displaying
        self.pyramid = pyrToVisualize
        self.keypoints = kpsToVisualize
        self.totOctaves = len(pyrToVisualize) if pyrToVisualize != None else None
        self.totScales = len(pyrToVisualize[0][0]) if pyrToVisualize != None else None
        self.startingSigma = -1

        # internal variables to realize pyramid navigation
        self.octaveToShow = 0
        self.scaleToShow = 0
        self.visualizeKeypoints = False
        self.gaussDogSelector = 0

        self.fig = None
        self.ax = None


    def _addCircle(self, ax, x, y, radius, color='r', fill=False):
        circle = plt.Circle((x, y), radius=radius, color=color, fill=fill)
        ax.add_artist(circle)

    def showKeypoints(self, img, alpha, title, keypoints, firstOctave, startingSigma, mode = 'non oriented'):
        """
        this method plots keypoints detected on an image
        inputs:
            - img: ndarray --> image numpy array
            - alpha: float --> value of alpha of image
            - title: string --> title of the plot
            - keypoints: listof lists --> keypoints structure organized in octaves
            - firstOctave: int --> either 0 or -1
            - startingSigma: float --> value of the starting sigma
            - mode: string --> either 'oriented' or 'non oriented' if orientation arrow needs to be displayed
        outputs:
        """
        plt.imshow(img, cmap=Greys_r, alpha=alpha)
        ax = plt.gca()
        plt.xlabel('width')
        plt.ylabel('height')
        plt.suptitle(title)


        addCircle = self._addCircle


        totScales = len(keypoints[0]) - 3
        octNum, kpNum, adjustement = 0, 0, 0
        for octave in keypoints:
            adjustement = 2**(octNum + firstOctave)
            for kp in octave:
                # the radius is proportional to the keypoint size adjusted, 2* to make it more visible
                r = 2*startingSigma*2**(kp.scale/totScales)*adjustement
                addCircle(ax, adjustement*kp.x, adjustement*kp.y, radius=r, color='g')

                if mode == 'oriented':
                    radAngle = radians(kp.orientation)
                    dx, dy = cos(radAngle), sin(radAngle)
                    ax.arrow(adjustement*kp.x,
                             adjustement*kp.y,
                             dx*r, dy*r,
                             head_width =.8 ,
                             head_length = .8,
                             width = .05,
                             fc='k',
                             ec='k').set_color('g')

                kpNum += 1
            octNum += 1
        plt.show()

    # this method defines a handler that uses the state of the class to decide what to show
    def _keyPressHandler(self, event):
        maxOctave = self.totOctaves
        maxScale = self.totScales + 3

        gauss = 0
        dog = 1

        # we cycle through every octave and scale
        if event.key == 'right':
            if self.scaleToShow + 1 < maxScale - self.gaussDogSelector:
                self.scaleToShow += 1
            else:
                self.scaleToShow = 0
                if self.octaveToShow + 1 < maxOctave:
                    self.octaveToShow += 1
                else:
                    self.octaveToShow = 0
        elif event.key == 'left':
            if self.scaleToShow - 1 >= 0:
                self.scaleToShow -= 1
            else:
                self.scaleToShow = maxScale - 1 - self.gaussDogSelector
                if self.octaveToShow - 1 >= 0:
                    self.octaveToShow -= 1
                else:
                    self.octaveToShow = maxOctave - 1
        elif event.key == 'up':
            if self.octaveToShow - 1 >= 0:
                self.octaveToShow -= 1
            else:
                self.octaveToShow = maxOctave - 1
        elif event.key == 'down':
            if self.octaveToShow + 1 < maxOctave:
                self.octaveToShow += 1
            else:
                self.octaveToShow = 0
        elif event.key == 'g':
            if self.gaussDogSelector != gauss:
                self.gaussDogSelector = gauss
        elif event.key == 'd':
            if self.gaussDogSelector != dog:
                self.gaussDogSelector = dog
            if self.scaleToShow > maxScale - 2:
                self.scaleToShow = maxScale - 2
        elif event.key == 'k':
            self.visualizeKeypoints = not self.visualizeKeypoints
        elif event.key == 'escape':
            plt.close()
        else:
            return

        plt.cla()

        if self.visualizeKeypoints:
            alpha = .5
        else:
            alpha = 1.
        plt.imshow(self.pyramid[self.octaveToShow][self.gaussDogSelector][self.scaleToShow], alpha=alpha, cmap=Greys_r)
        plt.xlabel('width')
        plt.ylabel('height')


        addCircle = self._addCircle


        kpNum = 0
        if self.visualizeKeypoints:
            for kp in self.keypoints[self.octaveToShow]:
                if round(kp.scale) == self.scaleToShow and kp.octave == self.octaveToShow:
                    kpNum += 1
                    r = 2*self.startingSigma*2**(kp.scale/self.totScales)
                    addCircle(self.ax, kp.x, kp.y, radius=r, color='g')
                    radAngle = radians(kp.orientation)
                    dx, dy = cos(radAngle), sin(radAngle)
                    self.ax.arrow(kp.x,
                             kp.y,
                             dx*r, dy*r,
                             head_width =.8 ,
                             head_length = .8,
                             width = .05,
                             fc='k',
                             ec='k').set_color('g')
            plt.suptitle('octave {0}, scale: {1}, number of keypoints: {2}'.format(self.octaveToShow, self.scaleToShow, kpNum), fontsize=15)
        else:
            plt.suptitle('octave {0}, scale: {1}'.format(self.octaveToShow, self.scaleToShow), fontsize=15)

        self.fig.canvas.draw()

    def navigatePyramid(self, pyramid, keypoints, startingSigma):
        """
        this method is a wrapper for the keypress handler and sets up its environment
        inputs:
            - pyramid: list of lists --> pyramid organized by octaves
            - keypoints: list of lists --> keypoints organized by octave
            - startingSigma: float --> starting sigma of sift detector
        """
        self.pyramid = pyramid
        self.keypoints = keypoints
        self.totOctaves = len(pyramid)
        self.totScales = len(pyramid[0][0]) - 3
        self.startingSigma = startingSigma

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self._keyPressHandler)

        plt.imshow(self.pyramid[0][0][0], cmap=Greys_r)
        plt.xlabel('width')
        plt.ylabel('height')
        plt.suptitle('octave 0, scale: 0', fontsize=15)
        plt.show()
