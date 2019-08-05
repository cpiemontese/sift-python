#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from numpy import zeros
from scipy.spatial import KDTree
from pprint import pprint
from matplotlib.cm import Greys_r

from sift_features_db.features_db import FeaturesDB
from auxiliary_modules.auxfuncs import exitWithErr

class SiftMatcher:

    def __init__(self, nnRatio = .8):
        self.nnRatio = nnRatio
        self.featuresDB = FeaturesDB()


    # ------------------------------------------------------------------------------------------------------------ #
    #                                 Keypoint Matching, section 7 of Lowe's paper                                 #
    # ------------------------------------------------------------------------------------------------------------ #


    # if db is a name we load it, if its not a list then we exit, if its a list we are happy and we return it
    def _checkAndLoadDB(self, db):
        if isinstance(db, str):
            if not self.featuresDB.loadDB(db):
                exitWithErr('dbLoadingError', 'Could not load database {0}'.format(db))
            else:
                print '{0} loaded'.format(db)
                return self.featuresDB.currentDB
        elif not isinstance(db, list):
            exitWithErr('dbTypeError', '{0} expected, {1} given instead'.format("<type 'list'>", type(db)))
        else:
            # it could be that db was serialized with serializeDescriptorsAndKeypoints
            # in that case we need to decode every image
            for i in xrange(len(db)):
                if '__ndarray__' in db[i]['Img']:
                    db[i]['Img'] = self.featuresDB.decodeNumpyArray(db[i]['Img'])
            return db


    def _match(self, featsListToMatch, featsListMatchedAgainst):
        # we are given lists of features like so: [{'Kp': ..., 'Descriptor': ...}, {'Kp':..., 'Descriptor': ...}, ...]
        # we extract the descriptors for both lists
        descListToMatch = [entry['Descriptor'] for entry in featsListToMatch]
        descListMatchedAgainst = [entry['Descriptor'] for entry in featsListMatchedAgainst]

        tree = KDTree(descListMatchedAgainst)   # we want to match list1 to list2 so we create a kdtree with list2 descriptors list
        matches = []                            # here we'll put every pair of matching keypoints

        query = tree.query
        nnRatio = self.nnRatio

        for i in xrange(len(descListToMatch)):
            dists, pos = query(descListToMatch[i], k=2)
            nn1, nn2 = dists[0], dists[1]

            if nn2 == .0 or nn1/nn2 < nnRatio:                      # in nn2 is .0 then nn1 is .0 too
                kp = featsListToMatch[i]['Kp']                      # ieth descriptor corresponds to the ieth keypoint
                matchedKp = featsListMatchedAgainst[pos[0]]['Kp']   # pos[0] === position of first nearest neighbor
                matches.append((kp, matchedKp, dists[0]))

        return matches

    def _createCompositeImage(self, imgToMatch, imgMatchedAgainst):
        imgToMatchRowsNum, imgToMatchColsNum = imgToMatch.shape
        imgMatchedAgainstRowsNum, imgMatchedAgainstColsNum = imgMatchedAgainst.shape

        compositeImgCols = imgToMatchColsNum+imgMatchedAgainstColsNum
        compositeImgRows = max(imgToMatchRowsNum, imgMatchedAgainstRowsNum)
        compositeImg = zeros((compositeImgRows, compositeImgCols))

        compositeImg[0:imgToMatchRowsNum, 0:imgToMatchColsNum] = imgToMatch
        compositeImg[0:imgMatchedAgainstRowsNum, imgToMatchColsNum:compositeImgCols] = imgMatchedAgainst

        return compositeImg

    def matchFeatures(self, dbToMatch, dbMatchedAgainst):
        # prendi in input o nomi di db da aprire o db già serializzati
        # per ogni immagine di un db fai il match con le immagini dell'altro, e plotta il match

        dbToMatch = self._checkAndLoadDB(dbToMatch)
        dbMatchedAgainst = self._checkAndLoadDB(dbMatchedAgainst)

        H = {}

        for entryToMatch in dbToMatch:
            for entryMatchedAgainst in dbMatchedAgainst:
                featsListToMatch = entryToMatch['SIFTFeatures']
                featsListMatchedAgainst = entryMatchedAgainst['SIFTFeatures']
                matches = self._match(featsListToMatch, featsListMatchedAgainst)

                # vota con la trasf di hough e fai model checking o come si chiama
                imgToMatch = entryToMatch['Img']
                imgMatchedAgainst = entryMatchedAgainst['Img']

                compositeImg = self._createCompositeImage(imgToMatch, imgMatchedAgainst)

                imgToMatchColsNum = imgToMatch.shape[1]
                for match in matches:
                    kp, matchedKp, dist = match
                    xCoords = [kp[0], matchedKp[0] + imgToMatchColsNum]
                    yCoords = [kp[1], matchedKp[1]]
                    plt.plot(xCoords, yCoords, linestyle='-', marker='o')

                    """
                    Work in Progress
                    dx, dy = matchedKp[0] - kp[0], matchedKp[1] - kp[1]
                    dscale = matchedKp[2]/kp[2]
                    dtheta = matchedKp[3] - kp[3]

                    k = str([dx, dy, dscale, dtheta])

                    if k in H:
                        H[k] += 1
                    else:
                        H[k] = 1

                    for k in H:
                    if H[k] >= 3:
                        pprint(k)
                pprint(H)"""

                plt.imshow(compositeImg, cmap=Greys_r)
                plt.show()
