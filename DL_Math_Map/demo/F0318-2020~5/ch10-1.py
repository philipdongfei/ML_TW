# -*- coding: utf-8 -*-
"""copy of ch10-deeplearning.ipynb
by makaishi2
"""
# 10章 深度學習數字辨識

# 必要 library 宣告
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # 忽略警告訊息

# PDF輸出用
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

"""## 10.7 程式實作 1
"""
# 讀入資料
import os
import urllib.request

mnist_file = 'mnist-original.mat'
mnist_path = 'mldata'
mnist_url = 'https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat'

# 資料內容確認
mnist_fullpath = os.path.join('.', mnist_path, mnist_file)
if not os.path.isfile(mnist_fullpath):
    # 資料下載
    mldir = os.path.join('.', 'mldata')
    os.makedirs(mldir, exist_ok=True)
    print("donwload %s started." % mnist_file)
    urllib.request.urlretrieve(mnist_url, mnist_fullpath)
    print("donwload %s finished." % mnist_file)

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home='.')

x_org, y_org = mnist.data, mnist.target


# 輸入資料處理

# step1 資料正規化, 令值的範圍為 [0, 1]
x_norm = x_org / 255.0

# 在最前面加入虛擬變數(1)
x_all = np.insert(x_norm, 0, 1, axis=1)

print('虛擬變數加入後的 shape ', x_all.shape)

# step 2 轉換成 One-hot-Vector

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_all_one = ohe.fit_transform(np.c_[y_org])
print('One Hot Vector 化後的 shape ', y_all_one.shape)

# step 3 訓練資料、驗證資料分割

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, y_train_one, y_test_one = train_test_split(
    x_all, y_org, y_all_one, train_size=60000, test_size=10000, shuffle=False)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, 
    y_train_one.shape, y_test_one.shape)

# 資料內容確認

N = 20
np.random.seed(123)
indexes = np.random.choice(y_test.shape[0], N, replace=False)
x_selected = x_test[indexes,1:]
y_selected = y_test[indexes]
plt.figure(figsize=(10, 3))
for i in range(N):
    ax = plt.subplot(2, N/2, i + 1)
    plt.imshow(x_selected[i].reshape(28, 28),cmap='gray_r')
    ax.set_title('%d' %y_selected[i], fontsize=16)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
