import cPickle
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import numpy as np
from sklearn.cross_validation import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.decomposition import PCA
import sklearn.feature_extraction.text as sktext
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import VarianceThreshold

import re
import random

import get_corpus

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


def make_W2V_features(X, tfidf=False):
    ''' 
    Create feature vectors for input sentences, by creating a word vector for 
    each word in a sentence, then averaging the vectors over sentences. Word 
    vectors are word-embeddings using model trained on Google News corpus.
    X: list of sentences
    '''

    # Load google model
    model = Word2Vec.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', 
                                             binary=True)
   
    # model = Word2Vec(size=300, min_count=1)
    # model.build_vocab(X)
    # model.intersect_word2vec_format('../data/GoogleNews-vectors-negative300.bin',
    #                                      binary=True)
    # model.train(X)

    # Method to create "averaged" sentence vector using model
    def buildWordVector(text, weights=None, vocabMap=None):
        # Vector size based on standard size in model
        # print text.shape, weights.shape

        size = 300
        vec = np.zeros(size).reshape((1, size))
        count = 0.

        words = re.split('\W+', text)

        for word in words:
            # Just some basic clean-up

            # word = word.lower()
            try:

                weight = 1.
                # Find tf-idf weight, if given
                try:
                    if weights is not None and vocabMap is not None:
                        idx = vocabMap.index(word)
                        weight = weights[0, idx]
                except:
                    # Potential stop-word or something else trivial
                    weight = 0.
                    # print "bad word or stop word?", word

                vec += (model[word].reshape((1, size))) * weight
                count += 1.
            except KeyError:
                continue

        # print vec.shape, weights.shape
        if count != 0:
            vec /= count
        return vec

    # Create vectors for each sentences and scale them
    if tfidf:

        transformer = sktext.TfidfVectorizer(min_df=1)

        weights = transformer.fit_transform(X)
        
        vocabMap = transformer.get_feature_names()

        vecs = np.concatenate([buildWordVector(X[z], weights[z], vocabMap) for z in range(len(X))])

    else:
        vecs = np.concatenate([buildWordVector(z) for z in X])
    vecs = scale(vecs)

    return vecs


def make_D2V_features(X, y):

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=100, test_size = 0.2)

    print sum(y_tr), len(y_tr), len(y_te)
    sentences = []

    for i in range(len(X_tr)):
        s = X_tr[i]

        sentences.append(LabeledSentence(s.split(),['TRAIN_%s' % i]))

    model = Doc2Vec(alpha=0.025, min_alpha=0.025, min_count=1, window=10, size=300, sample=1e-4, negative=5, workers=8)
    model.build_vocab(sentences)

    for epoch in range(10):
        model.train(sentences)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

    X_tr = [model.infer_vector(s.split()) for s in X_tr]

    sentences = []

    X_te = [model.infer_vector(s.split()) for s in X_te]

    # for v in model.docvecs:
    print model.docvecs

    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')

    clf.fit(X_tr, y_tr)
    print clf.score(X_tr, y_tr)
    print clf.score(X_te, y_te)
    # sentences = []
    # for i in range(len(X)):
    #     s = X[i]

    #     sentences.append(LabeledSentence(s.split(),['SENT_%s' % i]))

    # model = Doc2Vec(alpha=0.025, min_alpha=0.025, min_count=1, window=10, size=300, sample=1e-4, negative=5, workers=8)
    # model.build_vocab(sentences)

    # for epoch in range(10):
    #     model.train(sentences)
    #     model.alpha -= 0.002  # decrease the learning rate
    #     model.min_alpha = model.alpha  # fix the learning rate, no decay
   
    # return  model.docvecs


def process_IBC_data(demTrees, repTrees, getPhrase=True, minLen=15):
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


def run(corpus=1):

    print "Begin testing"
    
    if corpus == 0:
        # IBC corpus
        [demInput, repInput, neutral] = cPickle.load(open('../data/full_ibc/ibcData.pkl', 'rb'))
    
        demSents, repSents = process_IBC_data(demInput, repInput)

    else:
        # PSC corpus
        demSents, repSents = get_corpus.get_PSC()


    # Both data sets has less republican sentences, so even it out:
    demSents = random.sample(demSents, len(repSents))

    # Doc-2-vec testing (by sentence)

    # print "Testing Doc2Vec Features"
    
    # y = np.concatenate((np.ones(len(demSents)), np.zeros(len(repSents))))
    # X = make_D2V_features(demSents + repSents, y)

    # clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    

    # cv = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=15)

    # scores = cross_val_score(clf, X, y, cv=cv)
    
    # print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    print "Test BOW features on full sentences"

    # Create features and labels 
    y = np.concatenate((np.ones(len(demSents)), np.zeros(len(repSents))))
    X = make_BOW_features(np.concatenate((demSents, repSents)))

    # This classification model is based on earlier testing. Testing was 
    # limited to SVC method, but several kernels were tested, and parameters
    # were fitted with grid search with cross-validation.

    # clf = SVC(kernel='linear', C=1.0, gamma=0.1)
    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    
    cv = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=15)

    scores = cross_val_score(clf, X, y, cv=cv)    
    print scores
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    # # print "Test BOW features on full sentences + longer phrases"

    # # demPhrases, repPhrases = process_IBC_data(demInput, repInput, True)
    
    # # # This data has less republican sentences, so even it out:
    # # demPhrases = demPhrases[:len(repPhrases)]

    # # # Create features and labels 
    # # y2 = np.concatenate((np.ones(len(demPhrases)), np.zeros(len(repPhrases))))
    # # X2 = make_BOW_features(np.concatenate((demPhrases, repPhrases)))

    # # # Classification, again, based on earlier experimentation.
    # # #clf = SVC(kernel='rbf', C=1.0, gamma=0.1)
    # # clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=0.1)

    print "Test word2vec averaging over sentences"

    # Create features and labels 
    y = np.concatenate((np.ones(len(demSents)), np.zeros(len(repSents))))
    X = make_W2V_features(np.concatenate((demSents, repSents)))

    # Classification, based on earlier experimentation.
    # clf = SVC(kernel='rbf', C=1.0, gamma=0.01)
    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=.1)
    
    # Calculate scores with cross validation
    cv = StratifiedKFold(y, n_folds=10, shuffle=True, random_state=15)
    scores = cross_val_score(clf, X, y, cv=cv)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

    print "Test word2vec averaging + tfidf over sentences"

    # Create features and labels 
    y = np.concatenate((np.ones(len(demSents)), np.zeros(len(repSents))))
    X = make_W2V_features(np.concatenate((demSents, repSents)), True)

    # Classification, based on earlier experimentation.
    # clf = SVC(kernel='rbf', C=1.0, gamma=0.01)
    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=1.)
    
    # Calculate scores with cross validation
    cv = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=15)
    scores = cross_val_score(clf, X, y, cv=cv)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))



if __name__ == '__main__':

    # [demInput, repInput, neutral] = cPickle.load(open(
    #                                     '../data/full_ibc/ibcData.pkl', 'rb'))
    run(0) # 0 for IBC, 1 for PSC


