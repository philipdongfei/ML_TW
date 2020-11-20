# -*- coding: utf-8 -*-
"""ch07-regression.ipynb
bt makaishi2

# 7 章　線性迴歸
"""

# 宣告需要用到的程式庫
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# pdf 輸出
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

from sklearn.datasets import load_boston

# 準備訓練用的資料
boston = load_boston()
x_org, yt = boston.data, boston.target
feature_names = boston.feature_names
print('原data', x_org.shape, yt.shape)
print('項目名: ', feature_names)

# RM 資料整理
x_data = x_org[:,feature_names == 'RM']
print('整理後', x_data.shape)

# 在 x 向量第 1 個位置中加入虛擬變數
x = np.insert(x_data, 0, 1.0, axis=1)
print('加入虛擬變數後', x.shape)

# 印出輸入資料 x（含虛擬變數）
print(x.shape)
print(x[:5,:])

# 印出實際值 yt
print(yt[:5])

# 畫出資料散佈圖
plt.scatter(x[:,1], yt, s=10, c='b')
plt.xlabel('ROOM', fontsize=14)
plt.ylabel('PRICE', fontsize=14)
plt.show()

# 以預測函數 (1, x) 之值計算預測值 yp
def pred(x, w):
    return(x @ w)

# 初始化處理

# 資料樣本總數
M  = x.shape[0]

# 輸入資料之維數（含虛擬變數）
D = x.shape[1]

# 迭代運算次數
iters = 50000

# 學習率
alpha = 0.01

# 權重向量的初始值（預設全部為 1）
w = np.ones(D)

# 記錄評估結果用（僅記錄損失函數值）
history = np.zeros((0,2))

# 迭代運算
for k in range(iters):
    
    # 計算預測值（7.8.1）
    yp = pred(x, w)
    
    # 計算誤差（7.8.2）
    yd = yp - yt
    
    # 梯度下降法的實作（7.8.4）
    w = w - alpha * (x.T @ yd) / M
    
    # 繪製學習曲線所需資料之計算與儲存
    if ( k % 100 == 0):
        # 計算損失函數值（7.6.1）
        loss = np.mean(yd ** 2) / 2
        # 記錄計算結果
        history = np.vstack((history, np.array([k, loss])))
        # 顯示畫面
#        print( "iter = %d  loss = %f" % (k, loss))

# 損失函數的初始值、最終值
print('損失函數初始值: %f' % history[0,1])
print('損失函數最終值: %f' % history[-1,1])

# 計算繪製迴歸線所需之座標值
xall = x[:,1].ravel()
xl = np.array([[1, xall.min()],[1, xall.max()]])
yl = pred(xl, w)

# 繪製散佈圖與迴歸線
plt.figure(figsize=(6,6))
plt.scatter(x[:,1], yt, s=10, c='b')
plt.xlabel('ROOM', fontsize=14)
plt.ylabel('PRICE', fontsize=14)
plt.plot(xl[:,1], yl, c='k')
plt.show()

# 繪製學習曲線（第一組數除外）
plt.plot(history[1:,0], history[1:,1])
plt.show()
