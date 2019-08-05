#!/usr/bin/python
# -*- coding: utf-8 -*-

if __name__ == "__main__":

    from sift_detector import SiftDetector
    from scipy.ndimage.filters import gaussian_filter
    from scipy.ndimage.interpolation import rotate
    from scipy import misc
    from cv2 import GaussianBlur, IMREAD_GRAYSCALE
    from cv2.xfeatures2d import SIFT_create
    from numpy import array_equal, allclose
    from matplotlib.cm import Greys_r

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    imgName = '../imgs/house/houserot.jpg'
    img = cv2.imread(imgName, IMREAD_GRAYSCALE)
    #img = GaussianBlur(img, (0, 0), .5, .5)    # image has nominal sigma of .5

    mySift = SiftDetector(numberOfOctaves = 4)

    # Basic operations:
    mySift.createPyramidAndDoG(img)
    mySift.detectKeypoints(logUnfilteredKps=True)#, logUnorientatedKps=True)
    mySift.computeSIFTDescriptors()
    #mySift.printDescriptors()
    #mySift.printKpsStats()

    # DB operations
    mySift.saveDescriptorsToDB('../dbs/house.json')
    #mySift.matchKeypoints('../dbs/house.json')

    # Keypoints visualization:
    #mySift.showKeypoints()
    #mySift.showKeypoints(mode='unfiltered_keypoints')
    #mySift.showKeypoints(mode='unorientated_keypoints')

    #mySift.navigatePyramid()

    #"""
    from cv2.xfeatures2d import SIFT_create
    from pprint import pprint
    from math import floor, ceil

    oridiffs = []
    def findCommonKps(cv2Kps, myKps, unf=False):
        equalKps = set()
        for kp in cv2Kps:
            for ukp in myKps:
                kpx, kpy = kp[0]
                kpangle = kp[1]
                if unf:
                    kpxflr, kpyflr = floor(kpx), floor(kpy)
                    kpxceil, kpyceil = ceil(kpx), ceil(kpy)
                    ukpx, ukpy = ukp

                    flrflr = (kpxflr == ukpx and kpyflr == ukpy)
                    ceilceil = (kpxceil == ukpx and kpyceil == ukpy)
                    flrceil = (kpxflr == ukpx and kpyceil == ukpy)
                    ceilflr = (kpxceil == ukpx and kpyflr == ukpy)

                    if flrflr or ceilceil or flrceil or ceilflr:
                        equalKps.add(kp)
                else:
                    ukpx, ukpy = ukp[0]
                    if ((round(kpx) == round(ukpx)) or (kpx == ukpx)) and ((round(kpy) == round(ukpy)) or (kpy == ukpy)):
                        ukpori = ukp[1]
                        oridiffs.append(min(abs(kp[1] - ukpori), abs(kp[1] + 360. - ukpori), abs(kp[1] - ukpori - 360.)))
                        equalKps.add(kp)


        return equalKps

    def removeDups(kptolog, sift, unf=False):
        nonDups = set()
        if sift.FST_OCT == -1:
            for o in xrange(len(kptolog)):
                for kp in kptolog[o]:
                    x, y, s = kp.getCoords()
                    if unf:
                        nonDups.add((round(x*2.**(o-1)), round(y*2.**(o-1))))
                    else:
                        nonDups.add(((x*2.**(o-1), y*2.**(o-1)), 360. - kp.orientation))
        else:
            for o in xrange(len(kptolog) + sift.FST_OCT):
                for kp in kptolog[o]:
                    x, y, s = kp.getCoords()
                    if unf:
                        nonDups.add((round(x*2.**o), round(y*2.**o)))
                    else:
                        nonDups.add(((x*2.**o, y*2.**o, 360. - kp.orientation)))
        return nonDups

    nonDupKps = removeDups(mySift.keypoints, mySift)
    nonDupUnfKps = removeDups(mySift.unfilteredKps, mySift, unf=True)

    sift = SIFT_create()
    kps, des = sift.detectAndCompute(img, None)
    nonDupCv2Kps = set([(kp.pt, kp.angle) for kp in kps])

    kpsToCv2Commons = findCommonKps(nonDupCv2Kps, nonDupKps)
    unfToCv2Commons = findCommonKps(nonDupCv2Kps, nonDupUnfKps, unf=True)

    totalCommonKpsFound = unfToCv2Commons | kpsToCv2Commons

    #pprint(mySift.kpsDiscardingReasons)
    print 'num of keypoints also in opencv: {0}'.format(len(kpsToCv2Commons))
    print 'common: {0} on a total of {1} (con doppioni: {2})'.format(len(totalCommonKpsFound), len(nonDupCv2Kps), len(kps))

    from pprint import pprint
    from statistics import mean, variance
    #pprint(oridiffs)
    print 'variance:', variance(oridiffs) if len(oridiffs) > 1 else None, 'mean:', mean(oridiffs)
    #"""
    """
    plt.imshow(img, cmap=Greys_r, alpha=.5)
    maxY, maxX = img.shape
    for kp in nonDupKps:
        plt.plot(kp[0], kp[1], 'ro')
    plt.show()
    #"""
