#prepareDataset.py
#helper script that prepares our dataset for use by our RNN

#imports

import cPickle
import treeUtil
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

def extractWordsFromParseTree(givenPTree,vocabDict,vocabSet):
    #Function that extracts words from a given parse tree recursively
    #first check if it is a leaf (which is then a word)
    if isinstance(givenPTree,treeUtil.leafObj):
        #has a word
        if (givenPTree.word in vocabSet):
            #attach numerical value to it
            givenPTree.alpha = vocabDict[givenPTree.word]
        else : #need to add it to the set
            vocabSet.add(givenPTree.word)
            vocabDict[givenPTree.word] = len(vocabDict)
            #and then relabel the word
            givenPTree.alpha = vocabDict[givenPTree.word]
    else: #we have a node object, recurse through its child nodes
        extractWordsFromParseTree(givenPTree.c1,vocabDict,vocabSet)
        extractWordsFromParseTree(givenPTree.c2,vocabDict,vocabSet)

def generateVocabulary(givenDataset,vocabFilename,alteredDatasetFilename):
    vocabDict = {}
    vocabSet = set([]) #to make it easier to look up vocab words
    for parseTree in givenDataset:
        extractWordsFromParseTree(parseTree,vocabDict,vocabSet)
        #testing this out
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
