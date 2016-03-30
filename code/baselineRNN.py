# baselineRNN.py
# script designed to hold the functions to initially generate our RNN

# imports

import cPickle
import numpy as np
from structClass import Struct
import random #for SGD
import treeUtil
import copy #for help with keeping track of chain rule paths

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

def derivLanguageActivFunc(vec):
    #given a language vector, calculate the derivative function of the
    #laguage vector
    return (float(1) / np.cosh(vec)) ** 2

# neural network class


class neuralNet(Struct):
    def __init__(self, numLabels, sentenceDim, vocabSize, vocabDict,
                trainingSet):
        #for the softmax layer
        self.softmaxWeightMat = np.zeros((numLabels, sentenceDim))
        #for the basic language layer
        self.languageWeightMat = np.zeros((sentenceDim,sentenceDim))
        #have our word embedding matrix
        self.wordEmbedingMat = np.zeros((sentenceDim,vocabSize))
        self.vocabDict = vocabDict #to keep track of our vocabulary
        self.trainingSet = trainingSet #for training our data
        self.labelDict = self.setLabels(trainingSet)
        self.sentenceDim = sentenceDim
        self.weightsInitialized = False
        self.lossFunction = None
    
    def setLabels(self,trainingSet):
        #given a list of parse trees, create a label vector and assign to
        #each parse tree
        #first, get set of labels from training set
        labelDict = {}
        for i in xrange(len(trainingSet)):
            if (trainingSet[i].label not in labelDict):
                #add it in
                labelDict[trainingSet[i].label] = len(labelDict)
        #then get numpy to develop an identity matrix for this
        labelMatrix = np.identity(len(labelDict))
        #then attach label vectors to each parse tree
        for i in xrange(len(trainingSet)):
            #get column reference
            labelVecCol = labelDict[trainingSet[i].label]
            givenLabelVec = labelMatrix[:,labelVecCol]
            #then transpose to assign as column vector
            trainingSet[i].labelVec = np.array([givenLabelVec]).T
        return labelDict

    def vectorizeSentenceTree(self,sentenceTree):
        #given a parse tree, vectorize the parse tree
        if (isinstance(sentenceTree,treeUtil.leafObj)): #is a word,
            #look up in our word embeding matrix
            wordIndex = self.vocabDict[sentenceTree.word]
            wordVec = self.wordEmbedingMat[:,wordIndex]
            #then adjust it for column usage
            wordColumnVec = np.array([wordVec]).T #for transpose
            sentenceTree.langVec = wordColumnVec #for reference
            return wordColumnVec
        else: #we have a recursively defined object
            leftChildVec = self.vectorizeSentenceTree(sentenceTree.c1)
            rightChildVec = self.vectorizeSentenceTree(sentenceTree.c2)
            #calculate sentenceVec
            sentenceVec = languageActivFunc(
                    np.dot(self.languageWeightMat,leftChildVec)
                    + np.dot(self.languageWeightMat,rightChildVec))
            #assign it and then return
            sentenceTree.langVec = sentenceVec
            return sentenceVec

    def forwardProp(self, sentenceTree):
        # given a sentence vector of sentenceDim dimensions, output our
        # softmax layer
        if (self.weightsInitialized == False):
            #shoud initialize this
            self.initializedWeights()
        if (self.lossFunction == None):
            self.lossFunction = self.defaultLossFunction()
        #first vectorize sentence
        sentenceVec = self.vectorizeSentenceTree(sentenceTree)
        #then move the sentence through the softmax layer
        inputVec = np.dot(self.softmaxWeightMat, sentenceVec)
        givenSoftMaxVec = softMaxFunc(inputVec)
        return givenSoftMaxVec
    
    def predict(self, parseTree):
        #given the sentence vector, predicts the one-hot vector associated
        #with that sentence vector
        probabilityPredVec = self.forwardProp(parseTree)
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
    
        
    def findColGrad(self,givenSentenceTree,wordNum):
        #find the column gradient at column wordNum in the sentence tree
        if (isinstance(givenSentenceTree,treeUtil.leafObj)):
            #it is word, check if it of wordNum
            if (givenSentenceTree.alpha == wordNum):
                return np.ones((self.sentenceDim,1))
            else:
                return np.zeros((self.sentenceDim,1))
        else:
            #take gradient for a sentence
            outerLayerDeriv = derivLanguageActivFunc(
                            np.dot(self.languageWeightMat,
                                givenSentenceTree.c1.langVec
                                + givenSentenceTree.c2.langVec))
            #once we have outer layer, take inner layer to consider
            innerLayerDeriv = np.dot(self.languageWeightMat,
                    self.findColGrad(givenSentenceTree.c1,wordNum)
                    + self.findColGrad(givenSentenceTree.c2,wordNum))
            return outerLayerDeriv * innerLayerDeriv

    def buildWordEmbedingGradient(self,
                                givenSentenceTree,predictionVec,correctLabel):
        #helper for building our word embedding gradient
        softmaxLayerDeriv = np.dot((predictionVec - correctLabel).T,
                self.softmaxWeightMat).T
        #get column numbers for matrix
        columnNumList = []
        for leaf in givenSentenceTree.get_leaves():
            columnNumList.append(leaf.alpha) #contains column reference number
        columnNumList = list(set(columnNumList)) #to get unique
        wordEmbedingGradMatrix = np.random.rand(self.sentenceDim,
                                            len(self.vocabDict))
        #test purposes
        for columnNum in columnNumList:
            #find gradient for this column
            wordEmbedingGradMatrix[:,columnNum] = self.findColGrad(
                    givenSentenceTree,columnNum).T
        #then return structure
        return softmaxLayerDeriv * wordEmbedingGradMatrix
    
    #functions designed to find the language gradient
    
    def languageDerivRecursion(self,langGradientPath):
        #given a language gradient path (a list of nodeObj objects), create the 
        #language-level gradient with respect to this path
        assert(len(langGradientPath) >= 1)
        if (len(langGradientPath) == 1): #just need to take the derivative
            #with respect to the matrix
            givenPhrase = langGradientPath[0]
            functionInputVector = np.dot(self.languageWeightMat,
                                givenPhrase.c1.langVec + givenPhrase.c2.langVec)
            #take derivative at function level
            derivActivFuncOutput = derivLanguageActivFunc(functionInputVector)
            #by chain, take derivative wrt functionInputVector
            derivFunctionInputVector = (givenPhrase.c1.langVec 
                                            + givenPhrase.c2.langVec)
            return derivActivFuncOutput * derivFunctionInputVector
        else: #must take with respect to subsequent path
            givenPhrase = langGradientPath[0]
            functionInputVector = np.dot(self.languageWeightMat,
                                givenPhrase.c1.langVec + givenPhrase.c2.langVec)
            derivActivFuncOutput = derivLanguageActivFunc(functionInputVector)
            #take derivative wrt next phrase in the path
            currentPathOutputDeriv = (
                np.dot(derivActivFuncOutput.T,self.languageWeightMat)).T
            return currentPathOutputDeriv * self.languageDerivRecursion(
                    langGradientPath[1:])


    def getLanguageChainRulePaths(self,sentenceTree):
        #given a sentence tree, get a list of the gradient chain rule paths
        #to consider
        listOfChainRulePaths = []
        givenPath = [] #this is designed to keep track of our paths
        #to append to our list
        def getLanguageChainRulePathsWrapper(sentenceTree,listOfChainRulePaths,
                                            givenPath):
            #main function for finding a path dependent on
            if (not(isinstance(sentenceTree,treeUtil.leafObj))):
                #means that it is dependent on the language matrix
                givenPath.append(sentenceTree)
                listOfChainRulePaths.append(givenPath)
                #check if its left and right sides are dependent on the language
                #matrix
                if (not(isinstance(sentenceTree.c1,treeUtil.leafObj))):
                    leftGivenPath = copy.deepcopy(givenPath)
                    getLanguageChainRulePathsWrapper(sentenceTree.c1,
                            listOfChainRulePaths,leftGivenPath)
                if (not(isinstance(sentenceTree.c2,treeUtil.leafObj))):
                    rightGivenPath = copy.deepcopy(givenPath)
                    getLanguageChainRulePathsWrapper(sentenceTree.c2,
                            listOfChainRulePaths,rightGivenPath)
        #perform the wrapper
        getLanguageChainRulePathsWrapper(sentenceTree,listOfChainRulePaths,
                                        givenPath)
        return listOfChainRulePaths

    def buildLanguageWeightGradient(self,predictedLabel,correctLabel,
            givenSentenceTree):
        #main parent function that generates the gradient for the language
        #matrix given a sentence tree
        #first, account for the derivative at the softmax layer
        softmaxLayerDeriv = np.dot((predictedLabel - correctLabel).T,
                                    self.softmaxWeightMat)
        #then, generate the sentence level derivative by performing gradient
        #chain rule to all paths to the language level matrix
        listOfChainRulePaths = self.getLanguageChainRulePaths(givenSentenceTree)
        #then for each path, generate the language weight gradient based on that
        #path
        languageLayerDeriv = np.zeros((self.sentenceDim,1))
        for langGradientPath in listOfChainRulePaths:
            languageLayerDeriv += self.languageDerivRecursion(langGradientPath)
        languageWeightGradient = np.dot(softmaxLayerDeriv.T,
                                        languageLayerDeriv.T)
        return languageWeightGradient

    #main training algorithms

    def trainStochastically(self,numIterations,learningRate):
        #run SGD based on cross entropy function
        for i in xrange(numIterations):
            #get predictor ind
            givenSentenceTree = random.sample(self.trainingSet,1)[0]
            predictionVec = self.forwardProp(givenSentenceTree)
            #get gradient of weights
            correctLabel = givenSentenceTree.labelVec
            softmaxMatGradient = ((predictionVec - correctLabel)
                                    * givenSentenceTree.langVec.transpose())
            languageWeightGradient = self.buildLanguageWeightGradient(
                            predictionVec,correctLabel,givenSentenceTree)
            wordEmbedingGradient = self.buildWordEmbedingGradient(
                    givenSentenceTree,predictionVec,correctLabel)
            #then update weights
            self.softmaxWeightMat -= learningRate * softmaxMatGradient
            self.languageWeightMat -= learningRate * languageWeightGradient
            self.wordEmbedingMat -= learningRate * wordEmbedingGradient
            print self.getAccuracy(self.trainingSet)
    
    def trainManually(self,numIterations,learningRate):
        #helper that trains our neural network using standard GD (not
        #SGD)
        for i in xrange(numIterations):
            #initialize our gradients
            softmaxMatGradient = np.zeros(self.softmaxWeightMat.shape)
            languageWeightGradient = np.zeros(self.languageWeightMat.shape)
            wordEmbedingGradient = np.zeros(self.wordEmbedingMat.shape)
            #run through each parse tree
            for parseTree in self.trainingSet:
                predictionVec = self.forwardProp(parseTree)
                #add to gradient of weights
                correctLabel = parseTree.labelVec
                softmaxMatGradient += ((predictionVec - correctLabel)
                                    * parseTree.langVec.transpose())
                languageWeightGradient += self.buildLanguageWeightGradient(
                        predictionVec,correctLabel,parseTree)                
                #wordEmbedingGradient += self.buildWordEmbedingGradient(
                #            parseTree,predictionVec,correctLabel)
            #then update weights
            self.softmaxWeightMat -= learningRate * softmaxMatGradient
            self.languageWeightMat -= learningRate * languageWeightGradient
            #self.wordEmbedingMat -= learningRate * wordEmbedingGradient
            #print self.getAccuracy(self.trainingSet)
   
    def train(self,numIterations,learningRate,trainStochastically = False):
        #main layer to see method of training
        #check for initialization
        if (self.weightsInitialized == False):
            #initialize it
            self.initializedWeights()
        #then make training decision
        if (trainStochastically): #we will use the stochastic method
            self.trainStochastically(numIterations,learningRate)
        else:
            self.trainManually(numIterations,learningRate)

    #diagnostic methods

    def getAccuracy(self,parseTreeList):
        if (self.weightsInitialized == False):
            self.initializedWeights()
        #helper to get accuracy on a given set of data
        numCorrect = 0
        #check num correct
        for i in xrange(len(parseTreeList)):
            #get probability prediction,
            predictionVec = self.predict(parseTreeList[i])
            if (np.array_equal(predictionVec,parseTreeList[i].labelVec)):
                numCorrect += 1
        return float(numCorrect) / len(parseTreeList)

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

def testForwardPropagation(numLabels,sentenceDim,vocabFilename,datasetFilename):
    #tests out the forward propagation developed for our basic RNN
    #load vocabulary
    vocabDict = cPickle.load(open(vocabFilename,"rb"))
    #load dataset
    parseTreeList = cPickle.load(open(datasetFilename,"rb"))
    #then forward propagate through the neural network
    practiceNeuralNet = neuralNet(numLabels,sentenceDim,len(vocabDict),
                                    vocabDict,parseTreeList)
    print practiceNeuralNet.getAccuracy(practiceNeuralNet.trainingSet)
    practiceNeuralNet.train(80,1,True)


testForwardPropagation(3,6,"../data/ibcVocabulary.pkl",
                           "../data/alteredIBCData.pkl")
    
