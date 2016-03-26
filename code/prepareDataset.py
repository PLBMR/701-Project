#prepareDataset.py
#helper script that prepares our dataset for use by our RNN

#imports

import cPickle
#helpers

#main function

def collapseDataset(givenDataset):
    #assuming a dataset is a list of three indices, collapse the dataset
    #and dump it in the filename
    collapsedDataset = givenDataset[0]
    #collapse the three other indices
    collapsedDataset.extend(givenDataset[1])
    collapsedDataset.extend(givenDataset[2])
    return collapsedDataset

def generateVocabulary(givenDataset,vocabFilename,alteredDatasetFilename):
    vocabDict = {}
    vocabSet = set([]) #to make it easier to look up vocab words
    for parseTree in givenDataset:
        extractWordsFromParseTree(parseTree,vocabDict,vocabSet)
        #then test some aspects of the parse tree
    #then dump the altered dataset and vocabulary
    cPickle.dump(givenDataset,open(alteredDatasetFilename,"wb"))
    cPickle.dump(vocabDict,open(vocabFilename,"wb"))

def prepareDataset(ibcDatasetFilename,collapsedDatasetFilename,
                   vocabFilename):
    ibcDataset = cPickle.load(open(ibcDatasetFilename,"rb"))
    #first collapse the dataset, thankfully only has three indices
    ibcDataset = collapseDataset(ibcDataset)
    #then generate vocabulary while keeping track of indices within the vocab
    generateVocabulary(ibcDataset,vocabFilename,collapsedDatasetFilename)

prepareDataset("../data/full_ibc/ibcData.pkl","../data/alteredIBCData.pkl",
               "../data/ibcVocabulary.pkl")
