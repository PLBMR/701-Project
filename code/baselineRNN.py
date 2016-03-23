# baselineRNN.py
# script designed to hold the functions to initially generate our RNN

# imports

import cPickle
import numpy as np
from structClass import Struct
import random #for SGD

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
        def crossEntropy(outputY, targetY):
            # both the above come in as a list of lists
            assert(np.shape(outputY) == np.shape(targetY))
            np.sum(targetY * np.log(outputY))
        self.lossFunction = calculate_loss

    def train(self,numIterations,listOfLabels,listOfPredictors,learningRate):
        #helps train our weight matrix using SGD
        if (self.softMaxInitialized == False):
            #initialize it
            self.initializedWeights()
        predictorIndexList = range(len(listOfPredictors))
        random.shuffle(predictorIndexList)
        #run SGD based on cross entropy function
        for i in xrange(numIterations):
            #get predictor ind
            givenPredictorInd = predictorIndexList[i % len(predictorIndexList)]
            predictorVec = listOfPredictors[givenPredictorInd]
            predictionVec = self.forwardProp(predictorVec)
            #get gradient of weights
            correctLabel = listOfLabels[givenPredictorInd]

            weightMatGradient = ((predictionVec - correctLabel)
                                    * predictorVec.transpose())
            #then update weights
            self.softmaxWeightMat -= learningRate * weightMatGradient

# forward propagation

    # default loss function

# testing

def generateRandomVector(dimension):
    #helper to generate random vector
    randomVec = []
    for i in xrange(dimension):
        randomComp = [random.uniform(0,1)]
        randomVec.append(randomComp)
    randomVec = np.matrix(randomVec)
    return randomVec

def generateRandomLabel(numLabels):
    #helper to generate random labels
    #generate our random labels
    randomLabelList = []
    for i in xrange(numLabels):
        newLabelVec = []
        for j in xrange(numLabels):
            if (j == i):
                newLabelVec.append([1])
            else:
                newLabelVec.append([0])
        newLabelVec = np.matrix(newLabelVec)
        randomLabelList.append(newLabelVec)
    randomLabel = random.sample(randomLabelList,1)[0]
    return randomLabel

def testProcedure(numVectors,numIterations,learningRate):
    #tests out our training algorithm
    numLabels = 3
    vectorDim = 4
    practiceNN = neuralNet(numLabels,vectorDim)
    practiceNN.initializedWeights()
    print practiceNN.softmaxWeightMat
    #get our predictors and our labels
    listOfPredictors = []
    listOfLabels = []
    for i in xrange(numVectors):
        #generate new predictor
        listOfPredictors.append(generateRandomVector(vectorDim))
        listOfLabels.append(generateRandomLabel(numLabels))
    #see pre-train performance
    for i in xrange(numVectors)
    #then perform training algorithm
    practiceNN.train(numIterations,listOfLabels,listOfPredictors,learningRate)
    print practiceNN.softmaxWeightMat


testProcedure(30,2000,5)
