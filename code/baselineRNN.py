#baselineRNN.py
#script designed to hold the functions to initially generate our RNN

#imports

import cPickle
import theano

#load in dataset

def datasetLoadIn(datasetFilename):
    #helper designed to load in our initial dataset
    datasetFile = open(datasetFilename,"rb")
    #three lists of parse trees for different labels
    [liberalSent,conservSent,neutralSent] = cPickle.load(datasetFile)
    return [liberalSent,conservSent,neutralSent]

#activation functions

#forward propagation

def forwardPropagation(sentParseTree,wordEmbedMat,phraseLeftMat,phraseRightMat,
                       softMaxMat):
    #takes a given sentence (represented as a parse tree) and propagates it
    #through our neural network to give us our softmax probability vector


#generate RNN with dataset

#testing

print datasetLoadIn("../data/full_ibc/ibcData.pkl")[0][0].get_words()
