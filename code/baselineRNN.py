# baselineRNN.py
# script designed to hold the functions to initially generate our RNN

# imports

import cPickle
import numpy as np
from structClass import Struct
import random #for SGD
import treeUtil
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

def languageActivFunc(vec):
    #given a language vector, calculate the activation function of the langauge
    #vector
    return np.tanh(vec)

# neural network class


class neuralNet(Struct):
    def __init__(self, numLabels, sentenceDim, vocabSize, vocabDict):
        #for the softmax layer
        self.softmaxWeightMat = np.zeros((numLabels, sentenceDim))
        #for the basic language layer
        self.languageWeightMat = np.zeros((sentenceDim,sentenceDim))
        #have our word embedding matrix
        self.wordEmbedingMat = np.zeros((sentenceDim,vocabSize))
        self.vocabDict = vocabDict #to keep track of our vocabulary
        self.weightsInitialized = False
        self.lossFunction = None
    
    def vectorizeSentenceTree(sentenceTree):
        #given a parse tree, vectorize the parse tree
        if (isinstance(sentenceTree,treeUtil.leafObj)): #is a word,
            #look up in our word embeding matrix
            wordIndex = vocabDict[sentenceTree.alpha]
            wordVec = self.wordEmbedingMat[:,wordIndex]
            #then adjust it for column usage
            wordColumnVec = np.array([wordVec]).T #for transpose
            sentenceTree.set_label(wordColumnVec) #for reference
            return wordColumnVec
        else: #we have a recursively defined object
            leftChildVec = vectorizeSentenceTree(sentenceTree.c1).label
            rightChildVec = vectorizeSentenceTree(sentenceTree.c2).label
            #calculate sentenceVec
            sentenceVec = languageActivFunc(
                    np.dot(self.languageWeightMat,leftChildVec)
                    + np.dot(self.languageWeightMat,rightChildVec))
            #assign it and then return
            sentenceTree.set_label(sentenceVec)
            return sentenceVec

    def forwardProp(self, sentenceTree):
        # given a sentence vector of sentenceDim dimensions, output our
        # softmax layer
        if (self.weightsInitialized == False):
            #shoud initialize this
            self.initializedWeights()
        if (self.lossFunction == None):
            self.lossFunction = self.defaultLossFunctiona()
        #first vectorize sentence
        sentenceVec = self.vectorizeSentenceTree(sentenceTree)
        #then move the sentence through the softmax layer
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
        #helper for initializing our weights
        self.weightsInitialized = True
        self.softmaxWeightMat= np.random.rand(self.softmaxWeightMat.shape[0],
                                             self.softmaxWeightMat.shape[1])
        self.languageWeightMat = np.random.rand(self.languageWeightMat.shape[0],
                                             self.languageWeightMat.shape[1])
        self.wordEmbedingMat = np.random.rand(self.wordEmbedingMat.shape[0],
                                             self.wordEmbedingMat.shape[1])

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

            softmaxMatGradient = ((predictionVec - correctLabel)
                                    * predictorVec.transpose())
            #then update weights
            self.softmaxWeightMat -= learningRate * softmaxMatGradient

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

#test processses

def testForwardPropagation(numLabels,sentenceDim,vocabFilename,datasetFilename)
    
