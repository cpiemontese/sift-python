#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
This module implements a SIFT detector plus some additional features
useful to a better understanding of the internal workings of SIFT algorithm
"""


# This implementation is inspired to C/C++ implementations of SIFT that can be found here:
#       - https://github.com/robwhess/opensift/blob/master/src/sift.c
#       - https://gist.github.com/lxc-xx/7088609


import numpy as np
import matplotlib.pyplot as plt

from cv2 import GaussianBlur, subtract, resize, solve, cvtColor, fastAtan2
from cv2 import INTER_NEAREST, INTER_LINEAR, DECOMP_LU, COLOR_BGR2GRAY, CV_32F
from math import sin, cos, sqrt, atan, floor, degrees, radians, log, exp, pi
from matplotlib.cm import Greys_r
from numpy import dot, multiply, array, zeros
from numpy.linalg import inv, det
from scipy.ndimage import zoom
from scipy.spatial import KDTree
from scipy.ndimage.filters import gaussian_filter

from auxiliary_modules.auxfuncs import normalize, exitWithErr
from sift_features_db.features_db import FeaturesDB
from sift_matcher import SiftMatcher
from sift_datastructures.keypoint import Keypoint
from sift_visualizer import SiftVisualizer

from pprint import pprint

class SiftDetector:
    """
    SiftDetector class implements:
        - gaussian pyramid creation (createPyramidAndDoG)
        - keypoints detection and filtering (detectKeypoints)
        - keypoints' descriptors computation (computeSIFTDescriptors)

    SiftDetector also uses other classes to provide:
        - keypoints visualization
        - pyramid interactive navigation

    Its datastructures are organized like this:
        - self.pyramid is a list of pairs, one for every octave;
          each pairs contains a list of dogs and a list of smoothed images;
          there are totScales + 3 smoothed images and totScales+ 2 dogs for each pair
        - self.keypoints is a list of totOctaves lists of keypoints for every octave;
          each keypoints contains is an instance ok the Keypoint class
        - self.descriptors is a list of totOctaves lists of descriptors;
          the matching between keypoints and descriptors is 1 to 1, ie for each octave the ieth keypoint's descriptor is the ieth descriptor
    """
    def __init__(self,
                 startingSigma = 1.6,
                 numberOfOctaves = 4,           # 0 means that the number of octave computed will be as high as possible
                 numberOfScales = 3,
                 doubleStartingImage = True,
                 contrastThreshold = .04,
                 edgeThreshold = 10,
                 peakRatio = .8,
                 illuminationThreshold = .2):

        # ------- Data Structures ------- #
        self.img = []
        self.pyramid = []                           # list of numberOfOctaves octaves, contains [dogs, smoothed images]
        self.keypoints = []                         # list of keypoints organized by octaves
        self.unorientatedKps = []
        self.unfilteredKps = []
        self.descriptors = []

        # -- User modifiable variables -- #
        self.SIGMA_0 = startingSigma
        self.FST_OCT = -1 if doubleStartingImage else 0
        self.TOT_OCTS = numberOfOctaves if numberOfOctaves >= 0 else 0    # here we 'blindly' take the given value, we'll adjust it later
        self.TOT_SCLS = numberOfScales if numberOfScales >= 0 else 3
        self.EDGE_THRESH = edgeThreshold
        self.CONTR_THRESH = contrastThreshold
        self.PEAK_RATIO = peakRatio
        self.ILL_THRESH = illuminationThreshold

        # ----- Internal variables ----- #
        self.BORDER_OFST = 1
        self.MAX_INT_STEPS = 5                    # maximum number of interpolation steps, kinda arbitrary
        self.DESCR_HIST_WIDTH = 16
        self.kpsDiscardingReasons = {
            'discarding because: retval is not true': 0,
            'discarding because: out of borders': 0,
            'discarding because: interp not worked': 0,
            'discarding because: under threshold': 0,
            'discarding because: edge': 0,
        }

        # -- Internal auxiliay classes -- #
        self.featuresDB = FeaturesDB()
        self.featureMatcher = SiftMatcher()
        self.visualizer = SiftVisualizer()

        # --- Internal error messages --- #
        self.kperr = 'No keypoints, you need to detect them first (--> detectKeypoints())'
        self.pyrerr = 'Pyramid is empty, you need to compute it first (--> createPyramidAndDoG(img))'
        self.descerr = 'No descriptors to save, you need to compute them first (--> computeSIFTDescriptors())'


    # ----------------------------------------------------------------------------------------------------------- #
    #                                 Pyramid creation, section 3 of Lowe's paper                                 #
    # ----------------------------------------------------------------------------------------------------------- #


    def createPyramidAndDoG(self, img):
        """
        this method:
            creates a gaussian pyramid and relative DoGs given an image
        input:
            - img: numpy.ndarray --> image to create the gaussian pyramid of
            - doubleStartingImage: bool --> True or False wether starting image needs to be doubled or not
        """
        if type(img) is not np.ndarray:
            exitWithErr('wrongImageType', "Image needs to be <type 'numpy.ndarray'>, {0} given insted".format(type(img)))

        if len(img.shape) != 2:                     # image is not grayscale
            img = cvtColor(img, COLOR_BGR2GRAY)
        self.img = img
        img = np.float32(img)                       # we convert the image to float

        MAX_OCT_NUM = int(round(log(min(img.shape), 2)))
        log( MIN( init_img->width, init_img->height ) ) / log(2) - 2;
        self.TOT_OCTS = self.TOT_OCTS if self.TOT_OCTS > 0 else MAX_OCT_NUM
        self.TOT_OCTS = self.TOT_OCTS - self.FST_OCT

        if self.FST_OCT == -1:
            img = resize(img, (0, 0), fx=2., fy=2., interpolation=INTER_LINEAR)
            startingImageAssumedSigma = 1.
        else:
            startingImageAssumedSigma = .5

        sigma = self.SIGMA_0
        sigToApply = sqrt(sigma**2 - startingImageAssumedSigma**2)     # adjust sigma because of nominal smoothing
        img = GaussianBlur(img, (0, 0), sigToApply, sigToApply)        # this will be the first image of the first octave

        k = 2.**(1./self.TOT_SCLS)

        sig = [0.]*(self.TOT_SCLS + 3)
        sig[0] = sigma
        for j in xrange(1, self.TOT_SCLS + 3):
            sig_prev = pow(k, j - 1)*sigma
            sig_total = sig_prev*k
            sig[j] = sqrt(sig_total*sig_total - sig_prev*sig_prev)

        L, D = [], []                       # temporary arrays needed to memorize smoothed images and differences of gaussians
        self.pyramid = []                   # we reset the pyramid
        for i in xrange(self.TOT_OCTS):
            for j in xrange(self.TOT_SCLS + 3):
                if j == 0:
                    # the first image of every octave is img because img is either the image precomputed earlier, or a resized image
                    L.append(img)
                else:
                    # we do this because we are applying a filter to an already filtered image
                    # so the resulting image's sigma will be sqrt(sigma_applied^2 + previous_sigma^2)
                    # and in order to obtain the value we want we actually need to filter by this value
                    # L.append(GaussianBlur(L[j - 1], (0, 0), sig[j], sig[j]))
                    sigToObtain = sigma*(k**j)
                    sigToApply = sqrt(sigToObtain**2 - sigma**2)
                    L.append(GaussianBlur(img, (0, 0), sigToApply, sigToApply))
                    if j > 0:
                        D.append(subtract(L[j], L[j - 1]))

            sigma = 2*sigma
            # new image is the resizing of the image with gaussian smoothing of 2*sigma (2 images from the top of the pyramid)
            img = resize(L[self.TOT_SCLS], (0, 0), fx=.5, fy=.5, interpolation=INTER_NEAREST)
            self.pyramid.append([L, D])
            L, D = [], []

        print 'Pyramid created'


    # ------------------------------------------------------------------------------------------------------------- #
    #          Local extrema detection and Accurate keypoint localization, section 3 and 4 of Lowe's paper          #
    # ------------------------------------------------------------------------------------------------------------- #


    # derivatives are normalized to [0, 1]
    def _computeDerivatives(self, x, y, i, D):
        firstDerivativeScale = .5*(1./255.)
        dx = (D[i][y][x + 1] - D[i][y][x - 1])*firstDerivativeScale
        dy = (D[i][y + 1][x] - D[i][y - 1][x])*firstDerivativeScale
        ds = (D[i + 1][y][x] - D[i - 1][y][x])*firstDerivativeScale

        return [dx, dy, ds]

    def _computeHessian(self, x, y, i, D):
        secondDerivativeScale = 1./255.
        crossDerivativeScale = secondDerivativeScale*.25
        middleValue = D[i][y][x]

        dxx = (D[i][y][x + 1] - 2.*middleValue + D[i][y][x - 1])*secondDerivativeScale
        dyy = (D[i][y + 1][x] - 2.*middleValue + D[i][y - 1][x])*secondDerivativeScale
        dss = (D[i + 1][y][x] - 2.*middleValue + D[i - 1][y][x])*secondDerivativeScale
        dxy = (D[i][y + 1][x + 1] + D[i][y - 1][x - 1] - D[i][y + 1][x - 1] - D[i][y - 1][x + 1])*crossDerivativeScale
        dxs = (D[i + 1][y][x + 1] + D[i - 1][y][x - 1] - D[i + 1][y][x - 1] - D[i - 1][y][x + 1])*crossDerivativeScale
        dys = (D[i + 1][y + 1][x] + D[i - 1][y - 1][x] - D[i + 1][y - 1][x] - D[i - 1][y + 1][x])*crossDerivativeScale

        return [[dxx, dxy, dxs],
                [dxy, dyy, dys],
                [dxs, dys, dss]]

    # new vers: if maybeExtrema is negative we are checking if its a minima else we are checking for maxima
    # if maybeExtrema is positive we can't accept it as a maxima and if its negative it can't be a minima
    def _isLocalExtremum(self, x, y, scale, D, operator):
        maybeExtrema = D[scale][y][x]
        for i in xrange(-1, 2):
            for j in xrange(-1, 2):
                for s in xrange(-1, 2):
                    neighbor = D[scale + s][y + j][x + i]    # we assume that we have enough space and not go out of the image
                    if not operator(maybeExtrema, neighbor):
                        return False
        return True

    # this method returns the interpolated extremum on success and an empty list on failure
    def _filterLocalExtremum(self, x, y, scale, D):
        interpolatedExtremum = Keypoint(x, y, scale)
        interpolationWorked = False
        args = [x, y, scale, D]
        maxRows, maxCols = D[scale].shape

        _computeHessian = self._computeHessian
        _computeDerivatives = self._computeDerivatives
        border = self.BORDER_OFST
        totScales = self.TOT_SCLS

        for i in xrange(self.MAX_INT_STEPS):
            derivatives = array(_computeDerivatives(*args))
            hessian = array(_computeHessian(*args))
            hessDet = det(hessian)

            # X is the interpolated offset
            retval, X = solve(hessian, derivatives, flags=DECOMP_LU)
            if retval != True:
                self.kpsDiscardingReasons['discarding because: retval is not true'] += 1
                return None
            X = map(lambda x: -x, X)

            # if this is true then we don't need to interpolate anymore
            if abs(X[0]) < 0.5 and abs(X[1]) < 0.5 and abs(X[2]) < 0.5:
                interpolationWorked = True
                break

            interpolatedExtremum.addOffset(X)
            col, row, scl = interpolatedExtremum.getRoundedCoords()

            # we check that the new values make sense and if they do we can update the args, else this is not a keypoint
            if (col < border or col >= maxCols - border) or (row < border or row >= maxRows - border) or (scl < 1 or scl > totScales):
                self.kpsDiscardingReasons['discarding because: out of borders'] += 1
                return None
            else:
                args = [col, row, scl, D]

        if not interpolationWorked:
            self.kpsDiscardingReasons['discarding because: interp not worked'] += 1
            return None

        col, row, scl = interpolatedExtremum.getRoundedCoords()
        extremumValue = D[scl][row][col]/255.

        # we discard the keypoint if |D(X)| < contrastThreshold (in [0, 1] interval)
        if abs((extremumValue + .5*dot(derivatives, X)))*self.TOT_SCLS < self.CONTR_THRESH:
            self.kpsDiscardingReasons['discarding because: under threshold'] += 1
            return None

        # we compute the trace and determinant of 2x2 Hessian matrix
        # we descard the keypoint if their ratio is less than a threshold
        interpolatedExtremum.addOffset(X)
        args = interpolatedExtremum.getRoundedCoords() + D
        hessian = _computeHessian(*args)

        hessDet = hessian[0][0]*hessian[1][1] - hessian[0][1]**2
        hessTrace = hessian[0][0] + hessian[1][1]
        if hessDet <= 0. or (hessTrace**2)/hessDet >= ((self.EDGE_THRESH + 1.)**2)/self.EDGE_THRESH:
            self.kpsDiscardingReasons['discarding because: edge'] += 1
            return None

        return interpolatedExtremum

    def detectKeypoints(self, logUnfilteredKps = False, logUnorientatedKps = False):
        """
        this method:
            1) finds local extrema
            2) interpolates them using _filterLocalExtremum
            3) computes their orientation using _computeMainOrientations
        inputs:
            - logUnfilteredKps: bool --> True if the user wants unfiltered keypoints to be kept
            - logUnorientatedKps: bool --> True if the user wants unorientated (but filtered) keypoints to be kept
        """
        self.keypoints = []
        preliminaryContrastThreshold = floor(.5*(self.CONTR_THRESH/self.TOT_SCLS)*255.)

        if not self.pyramid:
            exitWithErr('noPyramidErr', self.pyrerr)


        # saved references ------------------------------------#
        selfKpAppend = self.keypoints.append
        selfUnfilteredKpsAppend = self.unfilteredKps.append
        selfUnorientatedKpsAppend = self.unorientatedKps.append
        _isLocalExtremum = self._isLocalExtremum
        _filterLocalExtremum = self._filterLocalExtremum
        _computeMainOrientations = self._computeMainOrientations
        # -----------------------------------------------------#


        for i in xrange(self.TOT_OCTS):
            D = self.pyramid[i][1]
            selfKpAppend([])
            if logUnfilteredKps:
                selfUnfilteredKpsAppend([])
            if logUnorientatedKps:
                selfUnorientatedKpsAppend([])


            # saved references ------------------------------------#
            selfKpIAppend = self.keypoints[i].append
            selfUnfilteredKpIAppend = None if not logUnfilteredKps else self.unfilteredKps[i].append
            selfUnorientatedKpIAppend = None if not logUnorientatedKps else self.unorientatedKps[i].append
            # -----------------------------------------------------#

            for j in xrange(1, self.TOT_SCLS + 1):    # we have s+2 dog for each octave, we "ignore" the first and the last
                height, width = D[j].shape

                for y in xrange(1, height - 2):
                    for x in xrange(1, width - 2):
                        val = D[j][y][x]

                        isMax = val > 0 and _isLocalExtremum(x, y, j, D, (lambda a, b: a >= b))
                        isMin = val < 0 and _isLocalExtremum(x, y, j, D, (lambda a, b: a <= b))
                        if abs(val) > preliminaryContrastThreshold and (isMax or isMin):
                            if logUnfilteredKps:
                                selfUnfilteredKpIAppend(Keypoint(x, y, j))

                            interpolatedExtremum = _filterLocalExtremum(x, y, j, D)

                            if not interpolatedExtremum:
                                continue
                            else:
                                interpolatedExtremum.octave = i
                                if logUnorientatedKps:
                                    selfUnorientatedKpIAppend(interpolatedExtremum)

                                # we can already compute the orientation histogram for this keypoint
                                mainOrientations = _computeMainOrientations(interpolatedExtremum)
                                mainOriLen = len(mainOrientations)
                                if mainOriLen == 36:
                                    continue

                                # we add a kp for each orientation we found
                                for o in xrange(mainOriLen):
                                    newOrientedKp = interpolatedExtremum.copy()
                                    newOrientedKp.orientation = mainOrientations[o]
                                    selfKpIAppend(newOrientedKp)

        print 'Keypoints detected, filtered and orientated'


    # ------------------------------------------------------------------------------------------------------------- #
    #                               Orientation assignment, section 5 of Lowe's paper                               #
    # ------------------------------------------------------------------------------------------------------------- #


    def _smoothHistogram(self, histogram):
        histLen = len(histogram)
        smoothedHistogram = [0.]*histLen

        # we extend the histogram so to ease the smoothing process
        histogram = [histogram[histLen - 2], histogram[histLen - 1]] + histogram + [histogram[0], histogram[1]]
        farBinsWeight = .0625       # 1/16
        closeBinsWeight = .25       # 4/16 (i.e. 1/4)
        currentBinWeight = .375     # 6/16 (i.e. 3/8)
        for i in xrange(histLen):
            farBinsWeightedValue = (histogram[i - 2] + histogram[i + 2])*farBinsWeight
            closeBinsWeightedValue = (histogram[i - 1] + histogram[i + 1])*closeBinsWeight
            smoothedHistogram[i] = farBinsWeightedValue + closeBinsWeightedValue + histogram[i]*currentBinWeight

        return smoothedHistogram

    def _interpolateHistogramPeak(self, precVal, currVal, succVal):
        dxx = precVal + succVal - 2.*currVal
        return (.5 * (precVal - succVal))/dxx if dxx != .0 else .0

    # this method computes an orientation histogram given a keypoint, its octave number, the size of the neighborhood and the number bins
    # NOTE: this method works for descriptor histogram too
    def _computeOrientationHistogram(self, kp, nbrdRadius, binsNum, windowSigma, gaussianScaleFactor, computingDescrHist = False):
        histogram = [0.] * binsNum
        binsToDegreesRatio = binsNum/360.               # how we divide the various angles into bins
        x, y, scale = kp.getRoundedCoords()
        L = self.pyramid[kp.octave][0][scale]           # we select the smoothed image at the right octave and scale
        yUpperBound, xUpperBound = L.shape

        if computingDescrHist:
            # if we are computing the histogram for the descriptor
            # we need to adjust the point to the angle of the keypoint
            cosOri, sinOri = cos(radians(kp.orientation))/nbrdRadius, sin(radians(kp.orientation))/nbrdRadius

        k = 0
        for i in xrange(-nbrdRadius, nbrdRadius + 1):
            for j in xrange(-nbrdRadius, nbrdRadius + 1):
                if computingDescrHist:
                    i, j = i*cosOri - j*sinOri, i*sinOri + j*cosOri

                xOfst, yOfst = int(round(x + i)), int(round(y + j))

                if (xOfst + 1 >= xUpperBound) or (yOfst + 1 >= yUpperBound) or (xOfst - 1 < 0) or (yOfst - 1 < 0):
                    continue

                xDiff = L[yOfst][xOfst + 1] - L[yOfst][xOfst - 1]
                yDiff = L[yOfst + 1][xOfst] - L[yOfst - 1][xOfst]

                weight = exp((i**2 + j**2)*gaussianScaleFactor)
                magnitude = sqrt(xDiff**2 + yDiff**2)
                orientation = fastAtan2(yDiff, xDiff)                # NOTE: ori is an anticlockwise angle

                # if we are computing the histogram for the descriptor we adjust the deteced orientation to achieve rotation independence
                if computingDescrHist:
                    orientation -= kp.orientation

                binNum = round(orientation*binsToDegreesRatio)
                if binNum >= binsNum:
                    binNum -= binsNum
                elif binNum < 0:
                    binNum += binsNum

                histogram[int(binNum)] += magnitude*weight
                k += 1

        if not computingDescrHist:
            #return histogram
            return self._smoothHistogram(histogram)
        else:
            return histogram

    # this method first computes the orientation histogram of a keypoint and then finds the main orientations (i.e. bin numbers)
    def _computeMainOrientations(self, kp):
        mainOrientations = []
        oriHistLen = 36

        # as per paper: sigma of weighted windows is 1.5 time that (i.e. the sigma) of the scale of the keypoint.
        windowSigma = 1.5*self.SIGMA_0*(2**(kp.scale/self.TOT_SCLS))    # 1.5*(1.6*k*2^(kp_scale))
        nbrdRadius = int(round(3.*windowSigma))                         # radius is proportional to the sigma we'll apply

        orientationHistogram = self._computeOrientationHistogram(kp,
                                                                 nbrdRadius,
                                                                 binsNum=oriHistLen,
                                                                 windowSigma=windowSigma,
                                                                 gaussianScaleFactor=-1./(2*(windowSigma**2)))
        margin = max(orientationHistogram)*self.PEAK_RATIO       # margin is 80% of th max orientation

        mainOrientationsAppend = mainOrientations.append
        _interpolateHistogramPeak = self._interpolateHistogramPeak

        for i in xrange(oriHistLen):
            prec = i - 1 if i > 0 else oriHistLen - 1
            succ = i + 1 if i < oriHistLen - 1 else 0
            iethValue = orientationHistogram[i]
            if iethValue >= margin and iethValue > orientationHistogram[prec] and iethValue > orientationHistogram[succ]:
                # we interpolate the peak in order to find a more exact orientation
                ofst = _interpolateHistogramPeak(orientationHistogram[prec], orientationHistogram[i], orientationHistogram[succ])
                b = i + ofst
                if b < 0.:
                    b += 36.
                if b >= 36.:
                    b -= 36.
                # we need to divide by the ratio between bins and degrees (i.e. 0.1) so we multiply by 10
                mainOrientationsAppend(b*10.)
        return mainOrientations


    # ------------------------------------------------------------------------------------------------------------- #
    #                               Local image descriptor, section 6 of Lowe's paper                               #
    # ------------------------------------------------------------------------------------------------------------- #


    def computeSIFTDescriptors(self):
        """
        this method:
            1) computes 16 orientation histograms (of radius 2) around specific points and concatenates them
            2) normalizes the descriptor obtained above and binds its values to be .2 max
            3) re-nomalizes the descriptor
        """
        self.descriptors = []

        if not self.keypoints:
            print self.kperr
            return

        # for each kp we need to compute 16 orientation histograms in a 16x16 area around the kp
        l = [-6, -2, 2, 6]

        points = [[x, y] for x in l for y in l]     # we generate all the possible points that are the center of our 4x4 areas


        # saved references ---------------------------------------------
        _computeOrientationHistogram = self._computeOrientationHistogram
        appendNewOctaveToDescriptor = self.descriptors.append
        startingSigma = self.SIGMA_0
        totScales = self.TOT_SCLS
        # --------------------------------------------------------------


        for o in xrange(len(self.keypoints)):
            appendNewOctaveToDescriptor([])

            appendNewDescriptorToCurrentOctave = self.descriptors[o].append
            currKpsOctave = self.keypoints[o]

            yMax, xMax = self.pyramid[o][0][0].shape
            for k in xrange(len(currKpsOctave)):
                appendNewDescriptorToCurrentOctave([])

                kp = currKpsOctave[k]
                kpSigma = startingSigma*(2**(kp.scale/totScales))
                for p in points:
                    tmpKp = kp.copy()
                    tmpKp.x += p[0]
                    tmpKp.y += p[1]
                    self.descriptors[o][k] += _computeOrientationHistogram(tmpKp,
                                                                           nbrdRadius=2,
                                                                           binsNum=8,
                                                                           windowSigma=8,
                                                                           gaussianScaleFactor=(-1./16.),
                                                                           computingDescrHist=True)

                self.descriptors[o][k] = normalize(self.descriptors[o][k])
                # any number greater than .2 is capped at .2 to achieve illumination independence
                self.descriptors[o][k] = map(lambda x: min(self.ILL_THRESH, x), self.descriptors[o][k])
                self.descriptors[o][k] = normalize(self.descriptors[o][k])

        print 'Descriptors computed'


    # ------------------------------------------------------------------------------------------------------------- #
    #                                               Various Functions                                               #
    # ------------------------------------------------------------------------------------------------------------- #


    def matchKeypoints(self, dBName):
        """
        this method:
            1) uses the featuresDB class to serialize detected features
            2) uses featureMatcher class to match detected features against a database
        """
        if not self.keypoints:
            exitWithErr('noKeypointsErr', 'No keypoints, you need to detect them')
        if not self.descriptors:
            exitWithErr('noDescErr', 'No descriptors, you need to compute them')

        jsonizedFeatures = self.featuresDB.serializeDescriptorsAndKeypoints(self.TOT_OCTS,
                                                                            self.FST_OCT,
                                                                            self.descriptors,
                                                                            self.keypoints,
                                                                            self.img)
        self.featureMatcher.matchFeatures([jsonizedFeatures], dBName)


    def saveDescriptorsToDB(self, dBName):
        """
        this method uses the featuresDB class to save detected features to a json file
        inputs:
            - dBName: string --> name of the database (ie json file) to save the data to
        """
        if not self.descriptors:
            print self.descerr
            return
        elif not self.keypoints:
            print self.kperr
            return

        if '.json' not in dBName:
            dBName = ''.join([dBName, '.json'])

        if self.featuresDB.saveToDB([self.TOT_OCTS, self.FST_OCT, self.descriptors, self.keypoints, self.img], dBName):
            print 'Descriptors saved to {0}'.format(dBName)
        else:
            print 'Could not save descriptors'


    def showKeypoints(self, alpha = .5, mode = 'actual_keypoints'):
        """
        this method uses visualizer class to show detected keypoints on an image
        inputs:
            - alpha: float --> alpha of the image
            - mode: string --> can be either 'actual_keypoints', 'unfiltered_keypoints', 'unorientated_keypoints';
                               it decides which keypoints will be shown
        """
        modeDict = {
            'actual_keypoints': {
                'keypoints': self.keypoints,
                'errMsg': self.kperr,
                'plotTitle': 'Keypoints'
            },
            'unfiltered_keypoints': {
                'keypoints': self.unfilteredKps,
                'errMsg': 'No unfiltered keypoints logged, you need to call detectKeypoints(logUnfilteredKps = True)',
                'plotTitle': 'Unfiltered Keypoints'
            },
            'unorientated_keypoints': {
                'keypoints': self.unorientatedKps,
                'errMsg': 'No unorientated keypoints logged, you need to call detectKeypoints(logUnorientatedKps = True)',
                'plotTitle': 'Unorientated Keypoints'
            }
        }

        keypoints = None
        if mode in modeDict:
            keypoints = modeDict[mode]['keypoints']
        else:
            print 'Mode {0} unknown'.format(mode)
            return

        if not keypoints:
            print modeDict[mode]['errMsg']
            return

        if mode == 'actual_keypoints':
            visMode = 'oriented'

        self.visualizer.showKeypoints(img=self.img,
                                      alpha=alpha,
                                      title=modeDict[mode]['plotTitle'],
                                      keypoints=modeDict[mode]['keypoints'],
                                      firstOctave=self.FST_OCT,
                                      startingSigma=self.SIGMA_0,
                                      mode=visMode)


    def navigatePyramid(self):
        """
        this method uses visualizer class to start pyramid navigation
        """
        self.visualizer.navigatePyramid(self.pyramid, self.keypoints, self.SIGMA_0)


    def printKpsStats(self):
        """
        this method prints keypoints' stats, i.e.:
            - number of keypoints (unorientated and unfiltered if they were logged)
            - number of keypoints per octave
        """
        octStats = []

        unoriNum = 0 if self.unorientatedKps else 'No logging done'
        unfiltNum = 0 if self.unfilteredKps else 'No logging done'
        actualNum = 0

        addNewOct = octStats.append

        for o in xrange(len(self.keypoints)):
            addNewOct([])
            currOctKpsLen = len(self.keypoints[o])
            octStats[o] = currOctKpsLen
            actualNum += currOctKpsLen

            if self.unfilteredKps:
                unfiltNum += len(self.unfilteredKps[o])
            if self.unorientatedKps:
                unoriNum += len(self.unorientatedKps[o])

        print '\nThis is how keypoints are distributed in octaves:'
        for o in xrange(len(octStats)):
            print '\t- octave {0}: {1} keypoints'.format(o, octStats[o])
        print 'Number of keypoints before filtering: {0}'.format(unfiltNum)
        print 'Number of keypoints before orientation assignment: {0}'.format(unoriNum)
        print 'Number of keypoints after filtering and orientation assignment: {0}'.format(actualNum)
