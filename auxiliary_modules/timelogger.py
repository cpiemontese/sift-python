#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
This module's public methods are:
    - startLogging(functionName)
    - endLogging(functionName)
    - printScoresStatistics()
    - printScores():
    - plotScores():
"""

import time
import matplotlib.pyplot as plt
from statistics import mean
from statistics import median
from statistics import variance


class TimeLogger():

    def __init__(self):
        self.scores = {}    # each entry has this layout: [[list of pending calls indexes], [calls times]]

    def _checkScoresExistence(self):
        if not self.scores:
            print "No scores to show"
            exit()

    def startLogging(self, funcName):
        if funcName not in self.scores:
            self.scores[funcName] = [[0], [time.time()]]
        else:
            self.scores[funcName][1].append(time.time())
            self.scores[funcName][0].append(len(self.scores[funcName][1]) - 1)

    def endLogging(self, funcName):
        if funcName not in self.scores:
            print "Error: You didn't start logging this function"
            return
        else:
            if len(self.scores[funcName][0]) == 0:
                print "Error: You didn't start logging this function"
                return
            currentCallNumber = self.scores[funcName][0].pop()
            self.scores[funcName][1][currentCallNumber] = time.time() - self.scores[funcName][1][currentCallNumber]

    def printScoresStatistics(self):
        self._checkScoresExistence()

        for key in self.scores:
            data = self.scores[key][1]
            var = "Couldn't calculate variance"
            if len(data) > 1:
                var = variance(data)
            print "function '{0}':\n\t mean: {1}\n\t median: {2}\n\t variance: {3}".format(key, mean(data), median(data), var)

    def printScores(self):
        self._checkScoresExistence()

        for key in self.scores:
            print key + "'s stats: {"
            print '\tthis function has been called: {0} times'.format(len(self.scores[key][1]))
            i = 1
            for time in self.scores[key][1]:
                print '\tcall {0} took: {1} seconds'.format(i, time)
                i += 1
            print '}'

    def plotScores(self):
        self._checkScoresExistence()

        for key in self.scores:
            y = self.scores[key][1]
            x = range(0, len(y))
            fig = plt.figure()
            plt.plot(x, y)
            fig.suptitle(key, fontsize = 15)
            plt.xlabel('number of calls', fontsize = 11)
            plt.ylabel('time', fontsize = 12)
            plt.show()
