from keras.layers import Input, Dense, recurrent
from keras.models import Model, Sequential
from keras.utils.np_utils import to_categorical


# this creates a model that includes
# the Input layer and three Dense layers
model = Sequential()
model.add(recurrent.LSTM(2,input_shape = (1,)))
model.add(Dense(2,activation = "softmax"))
model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
model.fit([1,2,3], to_categorical([0,1,1]))  # starts training
