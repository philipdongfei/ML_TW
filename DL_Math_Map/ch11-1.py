import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

D = 784

H = 128

num_classes = 10

from keras.datasets import mnist
(x_train_org, y_train), (x_test_org, y_test) \
    = mnist.load_data()

x_train = x_train_org.reshape(-1, D) / 255.0
x_test = x_test_org.reshape((-1,D)) / 255.0

from keras.utils import np_utils
y_train_ohe = \
    np_utils.to_categorical(y_train, num_classes)
y_test_ohe = \
    np_utils.to_categorical(y_test, num_classes)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(H, activation='relu', input_shape=(D,)))

model.add(Dense(H, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer = 'sgd',
              metrics = ['accuracy'])

batch_size = 512

nb_epoch = 50

history1 = model.fit(
    x_train,
    y_train_ohe,
    batch_size = batch_size,
    epochs = nb_epoch,
    verbose = 1,
    validation_data = (x_test, y_test_ohe))



