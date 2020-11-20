# -*- coding: utf-8 -*-
"""Copy of ch10-deeplearning.ipynb
by makaishi2
# 10章 深度學習數字辨識
"""

# 必要 library 宣告
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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

"""### 預測函數"""

# Sigmoid 函數
def sigmoid(x):
    return 1/(1+ np.exp(-x))

# Softmax 函數
def softmax(x):
    x = x.T
    x_max = x.max(axis=0)
    x = x - x_max
    w = np.exp(x)
    return (w / w.sum(axis=0)).T

"""### 評估函數"""

# 交叉商函數
def cross_entropy(yt, yp):
    return -np.mean(np.sum(yt * np.log(yp), axis=1))

# 評估處理 (準確率與損失函數)
from sklearn.metrics import accuracy_score

def evaluate(x_test, y_test, y_test_one, V, W):
    b1_test = np.insert(sigmoid(x_test @ V), 0, 1, axis=1)
    yp_test_one = softmax(b1_test @ W)
    yp_test = np.argmax(yp_test_one, axis=1)
    loss = cross_entropy(y_test_one, yp_test_one)
    score = accuracy_score(y_test, yp_test)
    return score, loss

"""### mini batch 處理"""

# mini batch 中取得 index 的類別 Index 
import numpy as np

class Indexes():
    
    # 建構初始化
    def __init__(self, total, size):
        # 總共要產生多少個值
        self.total   = total
        
        # 每次要從 indexes 中取出幾個值
        self.size    = size
        
        # indexes 是個要放入資料的 tuple 或 list, 
        # 初始化一開始裏面是空的
        self.indexes = np.zeros(0) 

    # 取出 indexes 中的資料, 每次取 batch size 個
    def next_index(self):
        # 預設為下次不需要再隨機產生新的 indexes 資料
        next_flag = False  
        
    # 若 indexes 內剩餘的資料量小於 batch size, 
    # 則重新隨機產生資料放入 indexes 中, 且每筆資料不重複
        if len(self.indexes) < self.size: 
            self.indexes = np.random.choice(self.total, 
                self.total, replace=False)
            next_flag = True
            
        # 如果 indexes 中的資料量還足夠, 
        # 則直接取出 batch size 個 
        index = self.indexes[:self.size]
        self.indexes = self.indexes[self.size:]
        
        # 將取出的 batch size 筆資料傳回, 並顯示下次是否需要
        # 再重新隨機產生新的 indexes 
        return index, next_flag




"""## 10.9 更換為 ReLU 函數"""

# ReLU 函數
def ReLU(x):
    return np.maximum(0, x)

# step 函數, ReLu 的微分
def step(x):
    return 1.0 * ( x > 0)

# 畫出 ReLu 與 step 函數圖形

xx =  np.linspace(-4, 4, 501)
yy = ReLU(xx)
plt.figure(figsize=(6,6))
#plt.ylim(0.0, 1.0)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)
plt.grid(lw=2)
plt.plot(xx, ReLU(xx), c='b', label='ReLU', linestyle='-', lw=3)
plt.plot(xx, step(xx), c='k', label='step', linestyle='-.', lw=3)
plt.xticks(size=14)
plt.yticks(size=14)
plt.legend(fontsize=14)
plt.show()



"""## 10.10 隱藏層有 2 層
"""

# 評估處理 (隱藏層有 2 層)
from sklearn.metrics import accuracy_score

def evaluate3(x_test, y_test, y_test_one, U, V, W):
    b1_test = np.insert(ReLU(x_test @ U), 0, 1, axis=1)
    d1_test = np.insert(ReLU(b1_test @ V), 0, 1, axis=1)
    yp_test_one = softmax(d1_test @ W)
    yp_test = np.argmax(yp_test_one, axis=1)
    loss = cross_entropy(y_test_one, yp_test_one)
    score = accuracy_score(y_test, yp_test)
    return score, loss

"""### 初始化處理"""

# 隱藏層節點數
H = 128
H1 = H + 1

# M: 訓練資料的總筆數
M  = x_train.shape[0]

# D: 輸入資料的維度
D = x_train.shape[1]

# N: 分類的類別數
N = y_train_one.shape[1]

# 機器學習的參數
alpha = 0.01
nb_epoch = 200
batch_size = 512
B = batch_size

# 權重矩陣 3 個, 初始化
U = np.random.randn(D, H) / np.sqrt(D / 2)
V = np.random.randn(H1, H) / np.sqrt(H1 / 2)
W = np.random.randn(H1, N) / np.sqrt(H1 / 2)

# 記錄評估結果 (損失函數值與準確率)
history4 = np.zeros((0, 3))

# 為 mini-batch 為訓練資料隨機建立索引
indexes = Indexes(M, batch_size)

# 記錄總共執行多少輪的變數
epoch = 0

"""### 主程式 4"""

while epoch < nb_epoch:
    # 選出要送去梯度計算的 mini-batch 資料
    index, next_flag = indexes.next_index()
    x, yt = x_train[index], y_train_one[index]
    
    # 預測值計算 (前饋式) 
    a = x @ U                        # (10.6.11)
    b = ReLU(a)                      # (10.6.12)
    b1 = np.insert(b, 0, 1, axis=1)  # 增加虛擬變數 
    c = b1 @ V                       # (10.6.13)
    d = ReLU(c)                      # (10.6.14)
    d1 = np.insert(d, 0, 1, axis=1)  # ダミー変数の追加
    u = d1 @ W                       # (10.6.15)   
    yp = softmax(u)                  # (10.6.16)
    
    # 誤差預測 
    yd = yp - yt                     # (10.6.17)
    dd = step(c) * (yd @ W[1:].T)    # (10.6.18)
    bd = step(a) * (dd @ V[1:].T)    # (10.6.19) 

    # 梯度計算
    W = W - alpha * (d1.T @ yd) / B  # (10.6.20)
    V = V - alpha * (b1.T @ dd) / B  # (10.6.21)
    U = U - alpha * (x.T @ bd) / B   # (10.6.22)

    if next_flag: # 1 epoch 結束的記錄
        score, loss = evaluate3(
            x_test, y_test, y_test_one, U, V, W)
        history4 = np.vstack((history4, 
            np.array([epoch, loss, score])))
#        print("epoch = %d loss = %f score = %f" 
#             % (epoch, loss, score))
        epoch = epoch + 1

"""### 結果確認"""

#損失函數值與準確率確認
print('初始狀態: 損失函數:%f 準確率:%f' 
     % (history4[1,1], history4[1,2]))
print('最終狀態: 損失函數:%f 準確率:%f' 
     % (history4[-1,1], history4[-1,2]))

# 學習曲線 (損失函數值)
plt.plot(history4[:,0], history4[:,1])
plt.ylim(0,2.5)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()

# 學習曲線 (準確率)
plt.plot(history4[:,0], history4[:,2])
plt.ylim(0,1)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()

# 資料確認
import matplotlib.pyplot as plt
N = 20
np.random.seed(123)
indexes = np.random.choice(y_test.shape[0], N, replace=False)

# x_org 選出來的 20 筆資料
x_selected = x_test[indexes]
y_selected = y_test[indexes]

# 預測值計算
b1_test = np.insert(ReLU(x_selected @ U), 0, 1, axis=1)
d1_test = np.insert(ReLU(b1_test @ V), 0, 1, axis=1)
yp_test_one = softmax(d1_test @ W)
yp_test = np.argmax(yp_test_one, axis=1)

# 畫出來
plt.figure(figsize=(10, 3))
for i in range(N):
    ax = plt.subplot(2, N/2, i + 1)
    plt.imshow(x_selected[i,1:].reshape(28, 28),cmap='gray_r')
    ax.set_title('%d:%d' % (y_selected[i], yp_test[i]),fontsize=14 )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()