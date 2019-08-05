#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
DescriptorsDB will:
    - save descriptors to a json file
    - read descriptors from a json file
    - other fancy stuff
"""


import sys
from pprint import pprint
from json import load, loads, dump, dumps

from auxiliary_modules.numpy_json_encoder import NumpyEncoder, json_numpy_obj_hook


class FeaturesDB:
    """
    DescriptorsDB class implements:
        - loading of a serialization from a database (ie a json file) (loadDB)
        - saving a serialization to a database (saveToDB)
        - serialization of descriptors and keypoints (serializeDescriptorsAndKeypoints)

    DescriptorsDB uses an external function json_numpy_obj_hook to provide image decoding from database
    """
    def __init__(self, dBName = None):
        self.currentDB = None
        self.numpyEncoder = NumpyEncoder()

        self.loadDB(dBName)     # if there's something to load we'll load it


    def loadDB(self, dBName):
        """
        this method:
            tries loading a json file given a name
        inputs:
            - dBName: string --> name of the file to load
        outputs:
            - bool --> True if file was loaded, False if not
        """
        dataFile = None

        if dBName is None:  # so not to print any error on init
            return False
        else:
            try:
                with open(dBName) as f:
                    if not self._isJson(f.read()):
                        return False
                    f.seek(0)
                    self.currentDB = loads(f.read())
                    for i in xrange(len(self.currentDB)):
                        self.currentDB[i]['Img'] = json_numpy_obj_hook(self.currentDB[i]['Img'])      # we decode every previously encoded image
                    return True
            except TypeError:
                print 'dBName must be either a string or a buffer, {0} passed instead'.format(type(dBName))
                return False
            except IOError as err:
                print 'error [{0}] occurred: {1} (filename {2})'.format(err.errno, err.strerror, dBName)
                return False


    def saveToDB(self, dataToSave, dBToSaveToName):
        """
        this method:
            1) loads data from the specified file if the file exists and it has data
            2) adds data to save to the file's data
            3) saves the file
            NOTE: if the file doesn't exist its created; this method will serialize data
        inputs:
            - dataToSave: array --> [totOctaves, firstOctave, descriptors, keypoints, img]
            - dBToSaveToName: string --> name of the file to save the data to
        output:
            - bool --> True if file was saved, False if it wasn't
        """
        dataToSave = self.serializeDescriptorsAndKeypoints(*dataToSave)
        try:                                        # we try to open the file if it exists
            with open(dBToSaveToName, 'r+') as f:
                jsonData = []
                try:
                    jsonData = load(f)
                except ValueError:                  # this exception will be thrown if f didn't contain any json
                    pass

                if isinstance(jsonData, list):
                    jsonData.append(dataToSave)
                else:
                    print 'Your json needs to be a list, {0} given instead'.format(type(jsonData))
                    return False

                f.seek(0)
                f.write(dumps(jsonData, indent=2))
                f.truncate()
                return True
        except IOError as e:    # if the file doesn't exist we just create it and save the data to it
            if e.errno == 2:
                with open(dBToSaveToName, 'w+') as f:
                    dump([dataToSave], f, indent=2)
                    return True


    def serializeDescriptorsAndKeypoints(self, totOctaves, firstOctave, descriptors, keypoints, img):
        """
        this method serializes to json sift keypoints and their relative descriptors
        inputs:
            - totOctaves: int --> number of octaves used for this instance of sift
            - firstOctave: int --> either 0 or -1
            - descriptors: numpy.ndarray --> array of arrays of 128 floats
            - keypoints: numpy.ndarray --> array of arrays of 4 floats representing a keypoint
            - image: numpy.ndarray --> image from which the keypoints where extracted
        output: list --> this list's format is described in jsonDataFormat file
        NOTE: keypoints are adjusted based on their octave
        """
        adjustement = 0
        # numpy arrays are not json serializable so we need to turn them into lists
        jsonSerialization = {'Img': self.numpyEncoder.default(img)}
        SIFTFeatures = []
        for o in xrange(totOctaves):
            adjustement = 2**(o + firstOctave)
            for kpNum in xrange(len(descriptors[o])):
                kp = keypoints[o][kpNum]
                adjustedKp = [kp.x*adjustement, kp.y*adjustement, kp.scale, kp.orientation, kp.octave]
                SIFTFeatures.append({'Kp': adjustedKp, 'Descriptor': descriptors[o][kpNum]})
        jsonSerialization['SIFTFeatures'] = SIFTFeatures

        return jsonSerialization


    def decodeNumpyArray(self, encodedArray):
        return json_numpy_obj_hook(encodedArray)

    # check if a string is json serialized
    def _isJson(self, maybeJsonData):
        try:
            loads(maybeJsonData)
        except ValueError:
            return False
        except TypeError:
            return False
        return True
