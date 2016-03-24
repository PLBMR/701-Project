# baselineRNN.py
# script designed to hold the functions to initially generate our RNN

# imports

import cPickle
import numpy as np
from structClass import Struct
import random  # for SGD


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
        self.WLeft = np.zeros((sentenceDim, sentenceDim))
        self.WRight = np.zeros((sentenceDim, sentenceDim))
        self.actFunction = None

    def forwardProp(self, sentenceVec):
        # given a sentence vector of sentenceDim dimensions, output our
        # softmax layer
        if (self.softMaxInitialized == False):
            # shoud initialize this
            self.initializedWeights()
        if (self.lossFunction == None):
            self.lossFunction = self.defaultLossFunction()
        inputVec = np.dot(self.softmaxWeightMat, sentenceVec)
        givenSoftMaxVec = softMaxFunc(inputVec)
        return givenSoftMaxVec

    def predict(self, sentenceVec):
        # given the sentence vector, predicts the one-hot vector associated
        # with that sentence vector
        probabilityPredVec = self.forwardProp(sentenceVec)
        # then produce one hot vector of this
        oneHotPredVec = np.zeros(probabilityPredVec.shape)
        predictedLabelIndex = np.argmax(probabilityPredVec)
        oneHotPredVec[predictedLabelIndex] = 1
        return oneHotPredVec

    def initializedWeights(self):
        self.softMaxInitialized = True
        self.softmaxWeightMat = np.random.rand(self.softmaxWeightMat.shape[0],
                                             self.softmaxWeightMat.shape[1])

    def setLossFunction(self, toSet):
        # sets the loss function
        self.lossFunction = toSet

    def defaultLossFunction(self):
        def crossEntropy(outputY, targetY):
            # both the above come in as a list of lists
            assert(np.shape(outputY) == np.shape(targetY))
            return (-1 * np.sum(targetY * np.log(outputY)))
        self.lossFunction = crossEntropy

    def setActivationFunction(self, toSet):
        # sets the activation function
        self.actFunction = toSet

    def defaultActivationFunction(self):
        self.actFunction = np.tanh

    def train(self, numIterations, listOfLabels, listOfPredictors, learningRate):
        # helps train our weight matrix using SGD
        if (self.softMaxInitialized == False):
            # initialize it
            self.initializedWeights()
        predictorIndexList = range(len(listOfPredictors))
        # run SGD based on cross entropy function
        for i in xrange(numIterations):
            # get predictor ind
            givenPredictorInd = random.sample(predictorIndexList, 1)[0]
            predictorVec = listOfPredictors[givenPredictorInd]
            predictionVec = self.forwardProp(predictorVec)
            # get gradient of weights
            correctLabel = listOfLabels[givenPredictorInd]

            weightMatGradient = ((predictionVec - correctLabel)
                                    * predictorVec.transpose())
            # then update weights
            self.softmaxWeightMat -= learningRate * weightMatGradient

    def getAccuracy(self, correctLabelList, predictorList):
        # helper to get accuracy on a given set of data
        assert(len(correctLabelList) == len(predictorList))
        numCorrect = 0
        # check num correct
        for i in xrange(len(predictorList)):
            # get probability prediction,
            vec = predictorList[i]
            predictionVec = self.predict(vec)
            if (np.array_equal(predictionVec, correctLabelList[i])):
                numCorrect += 1
        return float(numCorrect) / len(correctLabelList)

    def BPTS(self, topPtr):
        """
        sPtr: type: the head of the sentence level tree/ document level
        parse tree.

        puts in the final word-vec at the top level, so use topPtr.vector
        to access the final computation results

        Makes assumption that order doesn't matter. By order, we traverse
        all the way to the left first, then start backtracking from there
        assumes that we have the word vectors, but not the phrase/ sentence.
        Works on a full tree, and on a tree where the right val can be none

        @TODO:
        1) WLeft and WRight initialization,  dxd. Check initialization above
           composition matrix?
        2) add in the bias terms
        3) where are we getting x_w from? the d-dimensional vector
           from the sPtr, but what field in the class? and where
           does it come from? Anastassia?
        4) guarantees about structure? Always have a left node?
        5) fill in code to access the word-vec

        General questions with BPTS:
        1) based on fig.2, why is "so called climate change" a phrase
           but climate change isn't one
        2) does order matter in the BPTS?
        3) What does size of vocab mean?

        Further questions (not in scope of BPTS):
        1) where do we get W_e (word embeddings) from?
        2) where do we get w_cat from and what is it? Not explained in paper
        """
        rightFlag = True
        if self.actFunction == None:
            self.defaultActivationFunction()
        if topPtr.c1 == None and topPtr.c2 == None:
            return
        elif topPtr.c1 != None and topPtr.c2 == None:
            self.BPTS(topPtr.c1)
            rightFlag = False
        else:
            lVec = topPtr.c1.vector  # TODO: fill in with access to the vector
            if not(rightFlag):
                topPtr.vector = self.actFunction(self.WLeft * lVec)
                return
            rVec = topPtr.c2.vector  # TODO: fill in with access to the vector
            topPtr.vector = self.actFunction(self.WLeft *
                                             lVec + self.WRight * rVec)
            # TODO: fill in access to the vector
# testing


def generateRandomVector(dimension):
    # helper to generate random vector
    randomVec = []
    for i in xrange(dimension):
        randomComp = [random.uniform(0, 1)]
        randomVec.append(randomComp)
    randomVec = np.matrix(randomVec)
    return randomVec


def generateLabel(numLabels, predVec):
    # helper to generate random labels
    # generate our random labels
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

def testProcedure(numVectors, numIterations, learningRate):
    # tests out our training algorithm
    numLabels = 2
    vectorDim = 3
    practiceNN = neuralNet(numLabels, vectorDim)
    practiceNN.initializedWeights()
    print practiceNN.softmaxWeightMat
    # get our predictors and our labels
    listOfPredictors = []
    listOfLabels = []
    for i in xrange(numVectors):
        # generate new predictor
        givenRandomVec = generateRandomVector(vectorDim)
        listOfPredictors.append(givenRandomVec)
        listOfLabels.append(generateLabel(numLabels, givenRandomVec))
    # see pre-train performance
    print practiceNN.getAccuracy(listOfLabels, listOfPredictors)
    # then perform training algorithm
    practiceNN.train(numIterations, listOfLabels, listOfPredictors, learningRate)
    print practiceNN.softmaxWeightMat
    print practiceNN.getAccuracy(listOfLabels, listOfPredictors)

testProcedure(2000, 100000, 1)
