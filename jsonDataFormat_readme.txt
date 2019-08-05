    We assume that jsonData is a list of lists like this:

    [
        {
            'Img': normalImage,
            'SIFTFeatures': [
                {'Kp': ..., 'Descriptor': ...},
                {'Kp': ..., 'Descriptor': ...},
                ...,
            ]
        }
        {
            'Img': coolImage,
            'SIFTFeatures': [
                {'Kp': ..., 'Descriptor': ...},
                ...,
            ]
        }
    ]


    So a file will be considered empty if it is either one of these:
    1) actually empty
    2) an empty list []


    In either of the cases the data you generate from a sift detection will be appended to a main list of all data,
so to be able to discern an image and its descriptors from another image and its descriptors and so on...

    A note on images: in python an image is represented as a numpy array; unfortunately numpy arrays are not json serializable
by default, so the images arrays will be converted to a different format that needs to be decoded.
    On the bright side encoding is done by a class in 'numpy_json_encoder' package and decoding is done by a function,
provided in the same package, that can be found under 'auxiliary_modules'
