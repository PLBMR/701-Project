# baselineRNN.py
# script designed to hold the functions to initially generate our RNN

# imports

import cPickle
import numpy as np
from structClass import Struct


def datasetLoadIn(datasetFilename):
    # helper designed to load in our initial dataset
    datasetFile = open(datasetFilename, "rb")
    # three lists of parse trees for different labels
    [liberalSent, conservSent, neutralSent] = cPickle.load(datasetFile)
    return [liberalSent, conservSent, neutralSent]


# activation functions

def softMaxFunc(vec):
    # given a numpy matrix, calculate the softmax of that matrix
    softMaxVec = np.exp(vec) / np.sum(np.exp(vec))
    return softMaxVec

# neural network class


class neuralNet(Struct):
    def __init__(self, numLabels, sentenceDim):
        self.softmaxWeightMat = np.zeros((numLabels, sentenceDim))
        self.softMaxInitialized = False
        self.lossFunctionInitialized = False
        self.lossFunction = None

    def forwardProp(self, sentenceVec):
        # given a sentence vector of sentenceDim dimensions, output our
        # softmax layer
        if (self.softMaxInitialized == False):
            #shoud initialize this
            self.initializedWeights()
        inputVec = np.dot(self.softmaxWeightMat, sentenceVec)
        givenSoftMaxVec = softMaxFunc(inputVec)
        return givenSoftMaxVec

    def initializedWeights(self):
        self.softMaxInitialized = True
        self.softmaxWeightMat = np.ones(np.shape(self.softmaxWeightMat))

    def setLossFunction(self, toSet):
        #function
        self.lossFunctionInitialized = True
        self.lossFunction = toSet

    def defaultLossFunction(self):
        def calculate_loss(outputY, targetY):
            
        self.lossFunction = calculate_loss


#forward propagation

    # default loss function

# testing

idea = neuralNet(2,3)
print idea.predict(np.matrix([[1],[2],[3]]))
