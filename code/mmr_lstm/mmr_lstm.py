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

numExamples = 2
model = buildParLSTM(100,2,3,40)
x_train_b = np.random.random((numExamples, 5))
y_train = np.zeros((numExamples,2))

xtrain_a = np.array([[[1,2,3],[1,3]],[[2,3],[4,5]]])
#generate y_train
#for i in xrange(numExamples):
#   if (xtrain_a[i,0] < .5 and x_train_b[i,1] >= .5):
#        y_train[i,0] = 1
#   else:
#        y_train[i,1] = 1

print xtrain_a
print xtrain_a.shape
print x_train_b
print y_train
print y_train.shape
model.fit([xtrain_a,x_train_b],y_train,nb_epoch = 300)
