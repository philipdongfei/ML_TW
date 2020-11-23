from keras.models import Sequential
from keras.layers import Dense

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


batch_size = 512

nb_epoch = 50

#Sequential
model = Sequential()

#hidden layer1
model.add(Dense(H, activation='relu', kernel_initializer='he_normal', input_shape=(D,)))
#hidden layer2
model.add(Dense(H, activation='relu', kernel_initializer='he_normal'))

# output layer
model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))

model.compile(loss='categorical_crossentropy',
              optimizer = 'sgd',
              metrics=['accuracy'])

history1 = model.fit(
    x_train,
    y_train_ohe,
    batch_size = batch_size,
    epochs = nb_epoch,
    verbose = 1,
    validation_data = (x_test, y_test_ohe))


