from keras.layers import Input, Dense, Merge
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential
from keras.utils.np_utils import to_categorical
import numpy as np

# this creates a model that includes
# the Input layer and three Dense layers
def buildParLSTM(dataDim,numClasses,numSteps,layerSize):
    #helper for building an LSTM designed for two parallel sequences of
    #variable length
    #build A length
    encoderA = Sequential()
    encoderA.add(Embedding(dataDim,numSteps,mask_zero = True))
    encoderA.add(LSTM(layerSize))
    #build B length
    encoderB = Sequential()
    encoderB.add(Embedding(dataDim,numSteps,mask_zero = True))
    encoderB.add(LSTM(layerSize))
    #merge the two
    decoder = Sequential()
    decoder.add(Merge([encoderA,encoderB], mode = "concat"))
    #place softmax layer upon this
    decoder.add(Dense(numClasses, activation = "softmax"))
    decoder.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return decoder

model = buildParLSTM(3,2,3,2)
xtrain_a = np.random.random((3, 3))
x_train_b = np.random.random((3, 3))
y_train = np.random.randint(2,size = (3,2))
print xtrain_a
print x_train_b
print y_train
model.fit([xtrain_a,x_train_b],y_train)
