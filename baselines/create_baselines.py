import cPickle
from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.cross_validation import train_test_split, cross_val_score
import sklearn.feature_extraction.text as sktext
from sklearn.preprocessing import scale
from sklearn.svm import SVC

def make_BOW_features(X):
    '''
    Create feature vectors for the input list of sentences by using the tf-idf 
    value for each word that occurs in the sentence (idf calculated based on 
        entire input)
    X: list of sentences
    '''
    vectorizer = sktext.CountVectorizer(min_df=1)

    countX = vectorizer.fit_transform(X)

    transformer = sktext.TfidfTransformer()

    vecs = transformer.fit_transform(countX)
  
    return vecs


def make_W2V_features(X):
    ''' 
    Create feature vectors for input sentences, by creating a word vector for 
    each word in a sentence, then averaging the vectors over sentences. Word 
    vectors are word-embeddings using model trained on Google News corpus.
    X: list of sentences
    '''
    
    # Load google model
    model = Word2Vec.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin',
                                            binary=True)
   
    # Method to create "averaged" sentence vector using model
    def buildWordVector(text):
        # Vector size based on standard size in model
        size = 300
        vec = np.zeros(size).reshape((1, size))
        count = 0.

        words = text.split()
        for word in words:
            try:
                vec += model[word].reshape((1, size))
                count += 1.
            except KeyError:
                continue
        if count != 0:
            vec /= count
        return vec

    # Create vectors for each sentences and scale them
    vecs = np.concatenate([buildWordVector(z) for z in X])
    vecs = scale(vecs)

    return vecs


def process_IBC_data(demTrees, repTrees, getPhrase=False, minLen=15):
    '''
    Given the input of democratic/liberal and republican/conservative trees, 
    extract the underlying sentences (and in some cases phrases with sentences).
    demTrees, repTrees: tree representation of IBC sentences.
    getPhrase: whether to extract sub-sentences
    minLength: if extracting phrases, min length phrase to extract (if this gets 
        too small, labels will be noisey)
    ''' 
    demSent = []
    repSent = []

    # Only want one copy of each sentences (dups can happen in tree-traversal)
    allSent = []

    for treeSet in [demTrees, repTrees]:
        for tree in treeSet:
            if not getPhrase:
                # Only process root/whole sentence
                sent = tree.get_words()
                if tree.label == "Liberal":
                   demSent.append(sent)
                elif tree.label == "Conservative":
                   repSent.append(sent) 
            else:
                for node in tree:
                    # Only process subtrees with a label
                    if hasattr(node, 'label'):

                        sent = node.get_words()

                        if sent not in allSent and len(sent.split()) > minLen:
                            allSent.append(sent)

                            if node.label == "Liberal":
                                demSent.append(sent)
                            elif node.label == "Conservative":
                                repSent.append(sent)

    print "IBC phrase count:", len(demSent), len(repSent)
    return demSent, repSent


def runIBC():

    print "Begin IBC testing"
    
    [demInput, repInput, neutral] = cPickle.load(open('../data/full_ibc/ibcData.pkl', 'rb'))
    
    print "Test BOW features on full sentences"

    demSents, repSents = process_IBC_data(demInput, repInput)

    # This data has less republican sentences, so even it out:
    demSents = demSents[:len(repSents)]

    # Create features and labels 
    y = np.concatenate((np.ones(len(demSents)), np.zeros(len(repSents))))
    X = make_BOW_features(np.concatenate((demSents, repSents)))

    # This classification model is based on earlier testing. Testing was 
    # limited to SVC method, but several kernels were tested, and parameters
    # were fitted with grid search with cross-validation.

    clf = SVC(kernel='linear', C=1.0, gamma=0.1)

    scores = cross_val_score(clf, X, y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    
    print "Test BOW features on full sentences + longer phrases"

    demPhrases, repPhrases = process_IBC_data(demInput, repInput, True)
    
    # This data has less republican sentences, so even it out:
    demPhrases = demPhrases[:len(repPhrases)]

    # Create features and labels 
    y = np.concatenate((np.ones(len(demPhrases)), np.zeros(len(repPhrases))))
    X = make_BOW_features(np.concatenate((demPhrases, repPhrases)))

    # Classification, again, based on earlier experimentation.
    clf = SVC(kernel='rbf', C=1.0, gamma=0.1)

    # Calculate scores with cross validation
    scores = cross_val_score(clf, X, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print "Test word2vec averaging over sentences"

    # Create features and labels 
    y = np.concatenate((np.ones(len(demSents)), np.zeros(len(repSents))))
    X = make_W2V_features(np.concatenate((demSents, repSents)))

    # Classification, based on earlier experimentation.
    clf = SVC(kernel='rbf', C=1.0, gamma=0.0001)

    # Calculate scores with cross validation
    scores = cross_val_score(clf, X, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__ == '__main__':

    runIBC()

    [demInput, repInput, neutral] = cPickle.load(open(
                                        '../data/full_ibc/ibcData.pkl', 'rb'))

