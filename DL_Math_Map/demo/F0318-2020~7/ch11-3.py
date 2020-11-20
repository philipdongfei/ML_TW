# -*- coding: utf-8 -*-
"""ch11-keras.ipynb
@author: makaishi 
"""

# 必要 library 宣告
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# 資料準備
# 定義變數
# D：輸入層節點的維數
D = 784

# H：隱藏層節點的維數
H = 128

# 分類之類別數
num_classes = 10

# 以 Keras 的函數讀取資料
from keras.datasets import mnist
(x_train_org, y_train), (x_test_org, y_test) = mnist.load_data()

# 輸入資料處理成一維並正規化
x_train = x_train_org.reshape(-1, D) / 255.0
x_test = x_test_org.reshape(-1, D) / 255.0

# 實際值用 One-hot 編碼處理
from keras.utils import np_utils
y_train_ohe = np_utils.to_categorical(y_train, num_classes)
y_test_ohe = np_utils.to_categorical(y_test, num_classes)


"""## 11.5 優化的學習法

### SGD
"""

# 必要的 keras library
from keras.models import Sequential
from keras.layers import Dense

# 訓練小批量
batch_size = 512

# epoch 輪數
nb_epoch = 50

# Sequential model 定義
model = Sequential()

# 隱藏層1 定義
model.add(Dense(H, activation='relu', kernel_initializer='he_normal', input_shape=(D,)))

# 隱藏層2 定義
model.add(Dense(H, activation='relu', kernel_initializer='he_normal'))

# 輸出層
model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))

# 模型 compile
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'sgd',
              metrics=['accuracy'])

# 模型訓練
history1 = model.fit(
    x_train, 
    y_train_ohe,
    batch_size = batch_size, 
    epochs = nb_epoch,
    verbose = 1, 
    validation_data = (x_test, y_test_ohe))


"""### RmsProp"""

# Sequential 模型定義
model = Sequential()

# 隱藏層1 的定義
model.add(Dense(H, activation='relu', kernel_initializer='he_normal', input_shape=(D,)))

# 隱藏層2 的定義
model.add(Dense(H, activation='relu', kernel_initializer='he_normal'))

# 輸出層
model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))

# 模型 compile
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics=['accuracy'])

# 模型訓練
history2 = model.fit(
    x_train, 
    y_train_ohe,
    batch_size = batch_size, 
    epochs = nb_epoch,
    verbose = 1, 
    validation_data = (x_test, y_test_ohe))


"""### Momentum"""

# Sequential 模型的定義
model = Sequential()

# 隱藏層1 的定義
model.add(Dense(H, activation='relu', kernel_initializer='he_normal', input_shape=(D,)))

# 隱藏層2 的定義
model.add(Dense(H, activation='relu', kernel_initializer='he_normal'))

# 輸出層
model.add(Dense(num_classes, activation='softmax', kernel_initializer='he_normal'))

# 模型編譯
from keras import optimizers
sgd = optimizers.SGD(momentum = 0.9)
model.compile(loss = 'categorical_crossentropy',
              optimizer = sgd,
              metrics=['accuracy'])

# 模型訓練
history3 = model.fit(
    x_train, 
    y_train_ohe,
    batch_size = batch_size, 
    epochs = nb_epoch,
    verbose = 1, 
    validation_data = (x_test, y_test_ohe))


"""### 3 種曲線繪圖比較"""

#import matplotlib.pyplot as plt

# 學習曲線 (損失函數值)
plt.figure(figsize=(8,6))
plt.plot(history1.history['val_loss'],label='SGD', lw=3, c='k')
plt.plot(history2.history['val_loss'],label='rmsprop', lw=3, c='b')
plt.plot(history3.history['val_loss'],label='momentum', lw=3, c='k', linestyle='dashed')
plt.ylim(0,2)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.legend(fontsize=14)
plt.show()

import matplotlib.pyplot as plt

# 學習曲線 (準確率)
plt.figure(figsize=(8,6))
plt.plot(history1.history['val_accuracy'],label='SGD', lw=3, c='k')
plt.plot(history2.history['val_accuracy'],label='rmsprop', lw=3, c='b')
plt.plot(history3.history['val_accuracy'],label='momentum', lw=3, c='k', linestyle='dashed')
plt.ylim(0.8,1)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.legend(fontsize=14)
plt.show()