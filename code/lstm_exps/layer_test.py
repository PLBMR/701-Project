from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM

from keras.preprocessing.sequence import pad_sequences

import numpy as np
from random import sample

from wordEmbeds import get_data

# lib_words and rep_words are examples of phrases of either type.
# word vectors is a dict from words to word2vecs
lib_words, rep_words, word_vectors = get_data()


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

# Yes this method does not allow for testing, I am tired..
train = sample(range(len(X)), int(len(X)*0.8))

X_train = [X[i] for i in train]
Y_train = [Y[i] for i in train]

# Important change words into indicies
for i in range(len(X_train)):
	seq = X_train[i]

	new_seq = [np.zeros(n_symbols) for w in seq]

	for j in range(len(seq)):
		w = seq[j]
		new_seq[j][index_dict[w]] = 1

	X_train[i] = np.array(new_seq)
	Y_train[i] = np.array([Y_train[i]] * len(X_train[i]))

new_X_train = pad_sequences(X_train)
new_Y_train = pad_sequences(Y_train)
print new_X_train.shape
print new_Y_train.shape


# Now the actual model - 256/128 are random guesses for final dimensions...
model = Sequential()
model.add(Embedding(output_dim=300, input_dim=n_symbols, mask_zero=True, weights=[embedding_weights])) # note you have to put embedding weights in a list by convention
model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.train_on_batch(np.array(X_train[1]), np.array(Y_train[1]))

# model.fit(np.array(X_train), np.array(Y_train), batch_size=16, nb_epoch=5)

score = model.evaluate(X, Y, batch_size=16)