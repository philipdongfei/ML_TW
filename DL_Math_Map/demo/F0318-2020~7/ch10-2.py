# -*- coding: utf-8 -*-
"""ch10-deeplearning.ipynb
by makaishi2
"""

# 第10章 深度學習數字辨識
# 因應 Python 新版更新程式碼

# 必要 library 宣告
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # 忽略警告訊息

# PDF輸出用
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

"""## 10.7 程式實作 1
"""

# 讀入資料

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1,)

# 將讀入的資料指派給訓練資料

x_org, y_org = mnist.data, mnist.target.astype(np.int)

"""### 輸入資料整理"""

# step1 資料正規化, 令值的範圍在[0, 1]
x_norm = x_org / 255.0

# 在最前面加入虛擬變數 (1)
x_all = np.insert(x_norm, 0, 1, axis=1)

print('虛擬變數加入後的 shape', x_all.shape)

# step 2 轉換成 One-hot-Vector

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_all_one = ohe.fit_transform(np.c_[y_org])
print('One Hot Vector化後的 shape', y_all_one.shape)

# step 3 訓練資料、驗證資料分割

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, y_train_one, y_test_one = train_test_split(
    x_all, y_org, y_all_one, train_size=60000, test_size=10000, shuffle=False)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, 
    y_train_one.shape, y_test_one.shape)

# 資料內容確認

N = 20
np.random.seed(12)
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

# 評估處理 (準確率與損失函數值)
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

    # index取得関数    
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

# Indexes 類別測試

# 類別初始化
# 20: 總共有 20 筆資料
# 5: 每次取出 5 筆
indexes = Indexes(20, 5)

for i in range(6):
    # 呼叫 next_index 函數
    # 傳回值 1:  index 的 numpy 矩陣
    # 傳回值 2: 是否更新 index
    arr, flag = indexes.next_index()
    print(arr, flag)

"""### 初始化處理 (一)"""

# 變數宣告

# 隱藏層的節點數
H = 128
H1 = H + 1

# M: 訓練資料總數
M  = x_train.shape[0]
# D: 輸入資料的維數
D = x_train.shape[1]
# N: 分類的類別數
N = y_train_one.shape[1]

# 使用訓練資料多少輪 (epoch)
nb_epoch = 100
# mini-batch 的小批量
batch_size = 512
B = batch_size
# 學習率
alpha = 0.01

# 權重矩陣的元素預設皆為 1
V = np.ones((D, H))
W = np.ones((H1, N))

# 做結果評估的記錄 (損失函數值, 準確率)
history1 = np.zeros((0, 3))

# 為訓練資料隨機排序, 並取出 batch_size 個
indexes = Indexes(M, batch_size)

# 使用訓練資料的輪次計數
epoch = 0

"""### 主程式"""

# 主程式
while epoch < nb_epoch:

    # 由所有資料中取出小批量
    index, next_flag = indexes.next_index()
    x, yt = x_train[index], y_train_one[index]
    
    # 預測值計算 (前向運算) 
    a = x @ V                         # (10.6.3)
    b = sigmoid(a)                    # (10.6.4)
    b1 = np.insert(b, 0, 1, axis=1)   # 加入虛擬變數 
    u = b1 @ W                        # (10.6.5)   
    yp = softmax(u)                   # (10.6.6)
    
    # 誤差計算 
    yd = yp - yt                      # (10.6.7)
    bd = b * (1-b) * (yd @ W[1:].T)   # (10.6.8)

    # 梯度計算
    W = W - alpha * (b1.T @ yd) / B   # (10.6.9)
    V = V - alpha * (x.T @ bd) / B    # (10.6.10)
    
    # 記錄損失函數值與準確率
    if next_flag: # 1 epoch 結束後的記錄
        score, loss = evaluate(
            x_test, y_test, y_test_one, V, W)
        history1 = np.vstack((history1, 
            np.array([epoch, loss, score])))
        print("epoch = %d loss = %f score = %f" 
              % (epoch, loss, score))
        epoch = epoch + 1


"""### 結果確認 (一)"""


# 損失函數值與準確率
print('初始狀態: 損失函數:%f 準確率:%f' 
         % (history1[0,1], history1[0,2]))
print('最終狀態: 損失函數:%f 準確率:%f' 
         % (history1[-1,1], history1[-1,2]))

# 畫出學習曲線 (損失函數)
plt.plot(history1[:,0], history1[:,1])
plt.ylim(0,2.5)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()

# 畫出學習曲線 (準確率)
plt.plot(history1[:,0], history1[:,2])
plt.ylim(0,1)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()
