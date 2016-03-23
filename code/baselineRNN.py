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
        self.lossFunction = None

    def forwardProp(self, sentenceVec):
        # given a sentence vector of sentenceDim dimensions, output our
        # softmax layer
        if (self.softMaxInitialized == False):
            #shoud initialize this
            self.initializedWeights()
        if (self.lossFunction == None):
            self.lossFunction = self.defaultLossFunction()
        inputVec = np.dot(self.softmaxWeightMat, sentenceVec)
        givenSoftMaxVec = softMaxFunc(inputVec)
        return givenSoftMaxVec
    
    def predict(self, sentenceVec):
        #given the sentence vector, predicts the one-hot vector associated
        #with that sentence vector
        probabilityPredVec = self.forwardProp(sentenceVec)
        #then produce one hot vector of this
        oneHotPredVec = np.zeros(probabilityPredVec.shape)
        predictedLabelIndex = np.argmax(probabilityPredVec)
        oneHotPredVec[predictedLabelIndex] = 1
        return oneHotPredVec

    def initializedWeights(self):
        self.softMaxInitialized = True
        self.softmaxWeightMat= np.random.rand(self.softmaxWeightMat.shape[0],
                                             self.softmaxWeightMat.shape[1])

    def setLossFunction(self, toSet):
        #function
        self.lossFunctionInitialized = True
        self.lossFunction = toSet

    def defaultLossFunction(self):
        def crossEntropy(outputY, targetY):
            # both the above come in as a list of lists
            assert(np.shape(outputY) == np.shape(targetY))
            return (-1 * np.sum(targetY * np.log(outputY)))
        self.lossFunction = crossEntropy

    def train(self,numIterations,listOfLabels,listOfPredictors,learningRate):
        #helps train our weight matrix using SGD
        if (self.softMaxInitialized == False):
            #initialize it
            self.initializedWeights()
        predictorIndexList = range(len(listOfPredictors))
        #run SGD based on cross entropy function
        for i in xrange(numIterations):
            #get predictor ind
            givenPredictorInd = random.sample(predictorIndexList,1)[0]
            predictorVec = listOfPredictors[givenPredictorInd]
            predictionVec = self.forwardProp(predictorVec)
            #get gradient of weights
            correctLabel = listOfLabels[givenPredictorInd]

            weightMatGradient = ((predictionVec - correctLabel)
                                    * predictorVec.transpose())
            #then update weights
            self.softmaxWeightMat -= learningRate * weightMatGradient

    def getAccuracy(self,correctLabelList,predictorList):
        #helper to get accuracy on a given set of data
        assert(len(correctLabelList) == len(predictorList))
        numCorrect = 0
        #check num correct
        for i in xrange(len(predictorList)):
            #get probability prediction,
            vec = predictorList[i]
            predictionVec = self.predict(vec)
            if (np.array_equal(predictionVec,correctLabelList[i])):
                numCorrect += 1
        return float(numCorrect) / len(correctLabelList)

# testing

def generateRandomVector(dimension):
    #helper to generate random vector
    randomVec = []
    for i in xrange(dimension):
        randomComp = [random.uniform(0,1)]
        randomVec.append(randomComp)
    randomVec = np.matrix(randomVec)
    return randomVec

def generateLabel(numLabels,predVec):
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
    if (np.sum(predVec) >= 1.2):
        return randomLabelList[0]
    else:
        return randomLabelList[1]

def testProcedure(numVectors,numIterations,learningRate):
    #tests out our training algorithm
    numLabels = 2
    vectorDim = 3
    practiceNN = neuralNet(numLabels,vectorDim)
    practiceNN.initializedWeights()
    print practiceNN.softmaxWeightMat
    #get our predictors and our labels
    listOfPredictors = []
    listOfLabels = []
    for i in xrange(numVectors):
        #generate new predictor
        givenRandomVec = generateRandomVector(vectorDim)
        listOfPredictors.append(givenRandomVec)
        listOfLabels.append(generateLabel(numLabels,givenRandomVec))
    #see pre-train performance
    print practiceNN.getAccuracy(listOfLabels,listOfPredictors)
    #then perform training algorithm
    practiceNN.train(numIterations,listOfLabels,listOfPredictors,learningRate)
    print practiceNN.softmaxWeightMat
    print practiceNN.getAccuracy(listOfLabels,listOfPredictors)

testProcedure(2000,100000,1)
