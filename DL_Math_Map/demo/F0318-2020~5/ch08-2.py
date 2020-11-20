# -*- coding: utf-8 -*-
"""ch08-bi-classify.ipynb
by makaishi2
# Ch8　二元分類
"""

# 需要的函式庫
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# pdf 輸出用
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')



"""### 資料準備"""

# 訓練資料準備
from sklearn.datasets import load_iris
iris = load_iris()
x_org, y_org = iris.data, iris.target
print('原資料', x_org.shape, y_org.shape)

# 資料整理
#   分類 0, 1 
#   僅取用sepal_length與sepal_width兩項
x_data, y_data = iris.data[:100,:2], iris.target[:100]
print('目標資料', x_data.shape, y_data.shape)

# 加入虛擬變數 1
x_data = np.insert(x_data, 0, 1.0, axis=1)
print('加入虛擬變數後', x_data.shape)

# 　x_data, y_data 資料的 shape
print(x_data.shape, y_data.shape)
# 訓練資料、驗證資料分割 7比3
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=70, test_size=30, 
    random_state=123)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# 散佈圖顯示格式
x_t0 = x_train[y_train == 0]
x_t1 = x_train[y_train == 1]
plt.figure(figsize=(6,6))
plt.scatter(x_t0[:,1], x_t0[:,2], marker='x', c='b', label='0 (setosa)')
plt.scatter(x_t1[:,1], x_t1[:,2], marker='o', c='k', label='1 (versicolor)')
plt.xlabel('sepal_length', fontsize=14)
plt.ylabel('sepal_width', fontsize=14)
plt.xticks(size=16)
plt.yticks(size=16)
plt.legend(fontsize=16)
plt.show()

# 散佈圖打點
x_t0 = x_train[y_train == 0]
x_t1 = x_train[y_train == 1]
plt.figure(figsize=(6,6))
plt.scatter(x_t0[:,1], x_t0[:,2], marker='x', s=50, c='b', label='yt = 0')
plt.scatter(x_t1[:,1], x_t1[:,2], marker='o', s=50, c='k', label='yt = 1')
plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16)
plt.xticks(size=16)
plt.yticks(size=16)
plt.legend(fontsize=16)
plt.show()


# 設定訓練變數
x = x_train
yt = y_train

# 顯示輸入資料前 5 筆 (含虛擬變數)
print(x[:5])

# 實際值前 5 筆 
print(yt[:5])


# Sigmoid 函數定義
def sigmoid(x):
    return 1/(1+ np.exp(-x))

# 計算預測值
def pred(x, w):
    return sigmoid(x @ w)

"""### 評估"""

# 損失函數 (交差損失函數(交差熵函數)
def cross_entropy(yt, yp):
    # 交差熵函數計算
    ce1 = -(yt * np.log(yp) + (1 - yt) * np.log(1 - yp))
    # 交差熵平均值
    return(np.mean(ce1))

# 預測結果, 判斷是 1 或 0
def classify(y):
    return np.where(y < 0.5, 0, 1)

# 模型的評估函數 evaluate
from sklearn.metrics import accuracy_score

def evaluate(xt, yt, w):
    
    # 計算預測值
    yp = pred(xt, w)
    
    # 計算損失函數值
    loss = cross_entropy(yt, yp)
    
    # 預測值做分類 1 或 0
    yp_b = classify(yp)
    
    # 計算預測ˊ準確度
    score = accuracy_score(yt, yp_b)
    return loss, score


# 初始化處理

# 樣本數
M  = x.shape[0]
# x 資料的 shape (含虛擬變數)
D = x.shape[1]

# 迭代計算次數
iters = 10000

# 學習率
alpha = 0.01

# 權重向量的初始值
w = np.ones(D)

# 記錄損失函數的準確率歷史
history = np.zeros((0,3))

"""### 主程式 """

# 迭代運算開始

for k in range(iters):
    
    # 計算預測值 (8.6.1) (8.6.2)
    yp = pred(x, w)
    
    # 計算誤差 (8.6.4)
    yd = yp - yt
    
    # 梯度下降法公式 (8.6.6)
    w = w - alpha * (x.T @ yd) / M
    
    # 記錄用 (log)
    if ( k % 10 == 0):
        loss, score = evaluate(x_test, y_test, w)
        history = np.vstack((history, 
            np.array([k, loss, score])))
#        print( "iter = %d  loss = %f score = %f" 
#                % (k, loss, score))

"""### 結果確認"""

# 損失函數與準確度的確認
print('初始值: 損失函數:%f 準確率:%f' % (history[0,1], history[0,2]))
print('最終值: 損失函數:%f 準確率:%f' % (history[-1,1], history[-1,2]))

# 資料分佈圖, 分類成 1 與 0
x_t0 = x_test[y_test==0]
x_t1 = x_test[y_test==1]

# 決策邊界線用 x1 算出 x2 值
def b(x, w):
    return(-(w[0] + w[1] * x)/ w[2])
# 找出散佈圖上位於決策邊界上的最大 x1 與最小 x1
xl = np.asarray([x[:,1].min(), x[:,1].max()])
yl = b(xl, w)

plt.figure(figsize=(6,6))
# 輸出散佈圖
plt.scatter(x_t0[:,1], x_t0[:,2], marker='x', 
        c='b', s=50, label='class 0')
plt.scatter(x_t1[:,1], x_t1[:,2], marker='o', 
        c='k', s=50, label='class 1')
# 散佈圖格式設定
plt.plot(xl, yl, c='b')
plt.xlabel('sepal_length', fontsize=14)
plt.ylabel('sepal_width', fontsize=14)
plt.xticks(size=16)
plt.yticks(size=16)
plt.legend(fontsize=16)
plt.show()

# 損失函數的學習曲線
plt.figure(figsize=(6,4))
plt.plot(history[:,0], history[:,1], 'b')
plt.xlabel('iter', fontsize=14)
plt.ylabel('cost', fontsize=14)
plt.title('iter vs cost', fontsize=14)
plt.show()

# 準確率的學習曲線
plt.figure(figsize=(6,4))
plt.plot(history[:,0], history[:,2], 'b')
plt.xlabel('iter', fontsize=14)
plt.ylabel('accuracy', fontsize=14)
plt.title('iter vs accuracy', fontsize=14)
plt.show()