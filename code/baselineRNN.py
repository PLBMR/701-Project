# baselineRNN.py
# script designed to hold the functions to initially generate our RNN

# imports

import cPickle
import theano
import Struct from structClass.py

# load in dataset


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

    # train

    # loss function
    self.setLossFunction():
    

    # default loss function

# testing

print datasetLoadIn("../data/full_ibc/ibcData.pkl")[0][0].get_words()
