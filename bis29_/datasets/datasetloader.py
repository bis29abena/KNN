# import the necessary packages
import numpy as np
import cv2 as cv
import os


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are none initialise them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # initialise a list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract the class labels
            # assuming that our path has the following format
            # /path/to/dataset/{class}/image.jpg.
            image = cv.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # check to see if our preprocessor are not none
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our processed image as a feature vector
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)

            # show an update every verbose image
            if verbose > 1 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))

