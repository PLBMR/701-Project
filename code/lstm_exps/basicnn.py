from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM

from keras.preprocessing.sequence import pad_sequences

from sklearn.cross_validation import StratifiedKFold, train_test_split

import numpy as np
from random import sample
import numpy as np

from wordEmbeds import get_data3

# lib_words and rep_words are examples of phrases of either type.
# word vectors is a dict from words to word2vecs
lib_words, rep_words, word_vectors = get_data3()


# WordVec Prep Section
# Basically we make the word2vecs the initial weights the initial ones
# and the allow embed to train on for specific needs
vocab = word_vectors.keys()

# Map vocab words to unique indicies. We need an offset at 0.
index_dict = {val:(idx + 1) for idx,val in enumerate(vocab)}

vocab_dim = 300 # dimensionality of your word vectors
n_symbols = len(index_dict) + 1 # adding 1 to account for 0th index (for masking)

embedding_weights = np.zeros((n_symbols,vocab_dim))

# Set word vectors to this dict.
for word,index in index_dict.items():
    embedding_weights[index,:] = np.array(word_vectors[word])

print embedding_weights.shape

# Training set-up section
Y = [0] * len(lib_words) + [1] * len(rep_words)
X = lib_words + rep_words

print len(X), len(Y)

# make BOW-hot vectors
for i in range(len(X)):
	words = X[i]
	new_seq = np.zeros(n_symbols)

	for word in words:
		new_seq[index_dict[word]] = 1

	X[i] = np.array(new_seq)

X = np.array(X)
Y = np.array(Y)
print X.shape
print Y.shape

bias = np.random.uniform(-0.01, 0.01, 300)
print embedding_weights.shape
print bias.shape


# Try RMS prop??

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# model.fit(X, Y, batch_size=32, nb_epoch=40, validation_split=0.1)

cv = StratifiedKFold(Y, n_folds=5, shuffle=True, random_state=15)
for train_idx, test_idx in cv:

	model = Sequential()

	# Word2Vecs as start
	model.add(Dense(300, input_dim=n_symbols, weights=[embedding_weights, bias], activation='sigmoid'))
	
	# Standard uniform weights to start
	#model.add(Dense(64, input_dim=n_symbols, init='glorot_normal', activation='sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(64, activation='sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy',
	              optimizer='adagrad',
	              metrics=['accuracy'])


	X_train, X_test = X[train_idx], X[test_idx]
	y_train, y_test = Y[train_idx], Y[test_idx]

	model.fit(X_train, y_train, batch_size=32, nb_epoch=20, verbose=1, validation_data=(X_test, y_test))
	print "SCORE", model.evaluate(X_test, y_test)


