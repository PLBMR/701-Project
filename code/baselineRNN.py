# baselineRNN.py
# script designed to hold the functions to initially generate our RNN

# imports

import cPickle
import theano
import numpy as np
from structClass import Struct


def datasetLoadIn(datasetFilename):
    # helper designed to load in our initial dataset
    datasetFile = open(datasetFilename, "rb")
    # three lists of parse trees for different labels
    [liberalSent, conservSent, neutralSent] = cPickle.load(datasetFile)
    return [liberalSent, conservSent, neutralSent]


class neuralNet(Struct):


    # activation functions

    # forward propagation

    # generate RNN with dataset

    # initializeSoftMax layer

#activation functions

def softMaxFunc(vec):
    #given a numpy matrix

#neural network class

class neuralNet(Struct):
    def _init_(self,sentenceDim,numLabels):
        self.softmaxWeightMat = np.zeros((numLabels,sentenceDim))
        self.softMaxInitialized = False
        self.lossFunctionInitialized = False

    def forwardProp(self,sentenceVec):
        #given a sentence vector of sentenceDim dimensions, output our
        #softmax layer
            
                

    def prediction(self):


#forward propagation

    # default loss function

# testing

print datasetLoadIn("../data/full_ibc/ibcData.pkl")[0][0].get_words()
