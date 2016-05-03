import cPickle
from gensim.models.word2vec import Word2Vec
import numpy as np

# Extract the vocab that we want to use
vocabFilename = "../data/PSCVocabulary.pkl"
vocabDict = cPickle.load(open(vocabFilename,"rb"))

# Load google model
model = Word2Vec.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin',
                                     binary=True)

# Dummy set-up to get matrix rows in correct order
all_words = [[0.] * 300] * len(vocabDict.keys()) 

missing = 0
for word in vocabDict:
    idx = vocabDict[word]
    try:
    	# insert word, the splitting is a hack to increase matches
    	# with google news vocab.
        all_words[idx] = model[word.split('-')[0]]
    except:
        # If the google corpus does not have a word, just leave its
        # vector at zero, these comprise < 5 % of the data.
        # TODO: use more elegant alt solution
        missing += 1
        print word

print "Number of Words missing:", missing, len(all_words)

# Transpose to fit RNN needs
all_words = np.array(all_words).transpose()

with open("../data/PSCVocabMatrix.pkl", 'wb') as f:
	cPickle.dump(all_words, f)
