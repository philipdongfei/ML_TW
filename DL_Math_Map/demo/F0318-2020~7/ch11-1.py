# -*- coding: utf-8 -*-
"""
@author: makaishi 
"""


# 必要 library 宣告
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# PDF輸出用
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')


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

# 模型定義
# 載入必要的函式庫
from keras.models import Sequential
from keras.layers import Dense

# 定義 Sequential 模型
model = Sequential()

# 定義隱藏層1
model.add(Dense(H, activation ='relu', input_shape=(D, )))

# 定義隱藏層2
model.add(Dense(H, activation ='relu'))

# 輸出層
model.add(Dense(num_classes, activation ='softmax'))

# 編譯模型 
model.compile(loss = 'categorical_crossentropy',
           optimizer = 'sgd',     
           metrics = ['accuracy'])

# 訓練
# 訓練的批量
batch_size = 512

# 迭代運算次數
nb_epoch = 100

# 模型的訓練
history = model.fit(
    x_train, 
y_train_ohe, 
    batch_size = batch_size, 
    epochs = nb_epoch, 
    verbose = 1,
    validation_data = (x_test, y_test_ohe))
