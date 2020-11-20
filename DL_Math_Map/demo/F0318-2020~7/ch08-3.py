# -*- coding: utf-8 -*-
"""ch08-bi-classify.ipynb
by makaishi2
# Ch8　二元分類
"""

# 需要的函式庫
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# PDF輸出用
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

"""### 評価"""

# 損失函數 (交差熵函數)
def cross_entropy(yt, yp):
    # 交差熵函數計算
    ce1 = -(yt * np.log(yp) + (1 - yt) * np.log(1 - yp))
    # 交差熵平均值
    return(np.mean(ce1))

# 預測結果, 判斷是 1 或 0
def classify(y):
    return np.where(y < 0.5, 0, 1)

# 模型的評價函數 evaluate
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

# 損失函數與準確率的確認
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

# 匴失函數的學習曲線
plt.figure(figsize=(6,4))
plt.plot(history[:,0], history[:,1], 'b')
plt.xlabel('iter', fontsize=14)
plt.ylabel('cost', fontsize=14)
plt.title('iter vs cost', fontsize=14)
plt.show()

# 準確度的學習曲線
plt.figure(figsize=(6,4))
plt.plot(history[:,0], history[:,2], 'b')
plt.xlabel('iter', fontsize=14)
plt.ylabel('accuracy', fontsize=14)
plt.title('iter vs accuracy', fontsize=14)
plt.show()


""" 畫出 3D 圖 """
from mpl_toolkits.mplot3d import Axes3D
x1 = np.linspace(4, 7.5, 100)
x2 = np.linspace(2, 4.5, 100)
xx1, xx2 = np.meshgrid(x1, x2)
xxx = np.asarray([np.ones(xx1.ravel().shape), 
    xx1.ravel(), xx2.ravel()]).T
c = pred(xxx, w).reshape(xx1.shape)
plt.figure(figsize=(8,8))
ax = plt.subplot(1, 1, 1, projection='3d')
ax.plot_surface(xx1, xx2, c, color='blue', 
    edgecolor='black', rstride=10, cstride=10, alpha=0.1)
ax.scatter(x_t1[:,1], x_t1[:,2], 1, s=20, alpha=0.9, marker='o', c='b')
ax.scatter(x_t0[:,1], x_t0[:,2], 0, s=20, alpha=0.9, marker='s', c='b')
ax.set_xlim(4,7.5)
ax.set_ylim(2,4.5)
ax.view_init(elev=20, azim=60)

"""## scikit-learn 三種模型的比較"""

# 必要的函式庫
from sklearn.linear_model import LogisticRegression
from sklearn import svm

# 產生模型: 邏輯斯迴歸與 SVM
model_lr = LogisticRegression(solver='liblinear')
model_svm = svm.SVC(kernel='linear')

# 機器學習運算
model_lr.fit(x, yt)
model_svm.fit(x, yt)

# 線性迴歸
# 
lr_w0 = model_lr.intercept_[0]
# x1(sepal_length) 
lr_w1 = model_lr.coef_[0,1]
# x2(sepal_width) 
lr_w2 = model_lr.coef_[0,2]

# SVM
# 
svm_w0 = model_svm.intercept_[0]
# x1(sepal_length) 
svm_w1 = model_svm.coef_[0,1]
# x2(sepal_width) 
svm_w2 = model_svm.coef_[0,2]

# 找出決策邊界的座標, 代入 x1 找出 x2
def rl(x):
    wk = lr_w0 + lr_w1 * x
    wk2 = -wk / lr_w2
    return(wk2)

# 找出決策邊界的座標, 代入 x1 找出 x2
def svm(x):
    wk = svm_w0 + svm_w1 * x
    wk2 = -wk / svm_w2
    return(wk2)

y_rl = rl(xl)
y_svm = svm(xl)
# 結果確認
print(xl, yl, y_rl, y_svm)

# 畫出散佈圖邊界線
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
# 散佈圖打點
plt.scatter(x_t0[:,1], x_t0[:,2], marker='x', c='b')
plt.scatter(x_t1[:,1], x_t1[:,2], marker='o', c='k')
# 畫出三條決策邊界線
ax.plot(xl, yl, linewidth=2, c='k', label='Hands On')
# lr model
ax.plot(xl, y_rl, linewidth=2, c='k', linestyle="--", label='scikit LR')
# svm
ax.plot(xl, y_svm, linewidth=2, c='k', linestyle="-.", label='scikit SVM')

ax.legend()
ax.set_xlabel('$x_1$', fontsize=16)
ax.set_ylabel('$x_2$', fontsize=16)
plt.show()
