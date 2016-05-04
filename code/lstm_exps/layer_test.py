from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from random import sample
import numpy as np

from wordEmbeds import get_vectors

lib, rep = get_vectors()

Y = [0] * len(lib) + [1] * len(rep)

X = lib + rep

print len(X), len(Y)

stuff = range(len(X))
trainSize = int(len(X) * 0.8)
train = sample(stuff, trainSize)


X_train = [X[i] for i in train]
Y_train = [Y[i] for i in train]

X_train = np.array(X_train)
Y_train = np.array(Y_train)

model = Sequential()
# model.add(Embedding(300, ))
model.add(LSTM(output_dim=128, activation='sigmoid', inner_activation='hard_sigmoid', input_shape=(300,)))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=16, nb_epoch=10)
# score = model.evaluate(X_test, Y_test, batch_size=16)
