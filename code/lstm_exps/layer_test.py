from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM

import numpy as np
from random import sample
import numpy as np

from wordEmbeds import get_data

lib_words, rep_words, word_vectors = get_data()

# WordVec Prep Section
# Basically we make the word2vecs the initial weights the initial ones
# and the allow embed rain to train on for specific needs
vocab = word_vectors.keys()
# Map vocab words to unique indicies
index_dict = {val:(idx + 1) for idx,val in enumerate(vocab)}

vocab_dim = 300 # dimensionality of your word vectors
n_symbols = len(index_dict) + 1 # adding 1 to account for 0th index (for masking)
embedding_weights = np.zeros((n_symbols+1,vocab_dim))

print index_dict
for word,index in index_dict.items():
    embedding_weights[index,:] = word_vectors[word]

# Training set-up section
Y = [0] * len(lib_words) + [1] * len(rep_words)

X = lib_words + rep_words

# Yes this method does not allow for testing, I am tired..
train = sample(range(len(X)), int(len(X)*0.8))

X_train = [X[i] for i in train]
Y_train = [Y[i] for i in train]

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Important change words into indicies
for i in range(len(X_train)):
	seq = X_train[i]

	new_seq = [index_dict[w] for w in seq]

	#for j in range(len(seq)):
		#w = seq[j]
	#	new_seq[j][index_dict[w]] = 1

	X_train[i] = np.array(new_seq)

	print X_train[i]


# Now the actual model - 256/128 are random guesses for final dimensions...
model = Sequential()
model.add(Embedding(output_dim=300, input_dim=n_symbols + 1, mask_zero=True, weights=[embedding_weights])) # note you have to put embedding weights in a list by convention
model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=16, nb_epoch=10)
# score = model.evaluate(X_test, Y_test, batch_size=16)

for i in range(len(X_train)):
	if len(X_train[i]) ==  1:
		continue
	
	model.train_on_batch(np.array([X_train[i]]), np.array([Y_train[i]]))

score = model.evaluate(X, Y, batch_size=16)
