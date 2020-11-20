# -*- coding: utf-8 -*-
"""ch09-multi-classify.ipynb
by makaishi2
# 9章　多類別分類
"""


# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# pdf 輸出用
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')


# 準備訓練資料
from sklearn.datasets import load_iris
iris = load_iris()
x_org, y_org = iris.data, iris.target

# 輸出原始資料的 shape 與實際值的 shape
x_select = x_org[:,[0,2]]
print(' 原始 ', x_select.shape, y_org.shape)


# 散佈圖格式
x_t0 = x_select[y_org == 0]
x_t1 = x_select[y_org == 1]
x_t2 = x_select[y_org == 2]
plt.figure(figsize=(6,6))
plt.scatter(x_t0[:,0], x_t0[:,1], marker='x', c='k', s=50, label='0 (setosa)')
plt.scatter(x_t1[:,0], x_t1[:,1], marker='o', c='b', s=50, label='1 (versicolour)')
plt.scatter(x_t2[:,0], x_t2[:,1], marker='+', c='k', s=50, label='2 (virginica)')
plt.xlabel('sepal_length', fontsize=14)
plt.ylabel('petal_length', fontsize=14)
plt.xticks(size=14)
plt.yticks(size=14)
plt.legend(fontsize=14)
plt.show()



# 加入虛擬變數
x_all = np.insert(x_select, 0, 1.0, axis=1)

# y 換換為 one-hot-向量
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False,categories='auto')
y_work = np.c_[y_org]
y_all_one = ohe.fit_transform(y_work)
print('原始資料的 shape', y_org.shape)
print('2 維化', y_work.shape)
print('One Hot 向量化後的 shape', y_all_one.shape)

# 分割出訓練資料與驗證資料各一半
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test, y_train_one, y_test_one = train_test_split(
    x_all, y_org, y_all_one, train_size=75, test_size=75, random_state=123)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, 
    y_train_one.shape, y_test_one.shape)

print('輸入資料(x)')
print(x_train[:5,:])

print('實際值(y)')
print(y_train[:5])

print('實際值 (One Hot 編碼後)')
print(y_train_one[:5,:])



# 訓練對象
x, yt  = x_train, y_train_one


"""### 預測函數 """

# softmax 函數 (9.7.3)
def softmax(x):
    x = x.T
    x_max = x.max(axis=0)
    x = x - x_max
    w = np.exp(x)
    return (w / w.sum(axis=0)).T

# 計算預測值 (9.7.1, 9.7.2)
def pred(x, W):
    return softmax(x @ W)

"""### 評估 """

# 交叉熵函數 (9.5.1)
def cross_entropy(yt, yp):
    return -np.mean(np.sum(yt * np.log(yp), axis=1))

# 模型評估函數
from sklearn.metrics import accuracy_score

def evaluate(x_test, y_test, y_test_one, W):
    
    # 計算預測值機率
    yp_test_one = pred(x_test, W)
    
    # 由機率預務 1, 2, 3
    yp_test = np.argmax(yp_test_one, axis=1)
    
    # 計算損失函數值
    loss = cross_entropy(y_test_one, yp_test_one)
    
    # 計算準確率
    score = accuracy_score(y_test, yp_test)
    return loss, score


# 初始化處理

# 樣本數
M  = x.shape[0]
# 輸入資料的 shape
D = x.shape[1]
# 分類的類別數
N = yt.shape[1]

# 迭代運算次數
iters = 10000

# 學習率
alpha = 0.01

# 權重矩陣初始化, 所有元素為 1
W = np.ones((D, N)) 

# 評価結果記録用
history = np.zeros((0, 3))

"""### 主程式 """

#  主程式
for k in range(iters):
    
    # 計算預測值 (9.7.1)　(9.7.2)
    yp = pred(x, W)
    
    # 計算誤差 (9.7.4)
    yd = yp - yt

    # 更新權重矩陣 (9.7.5)
    W = W - alpha * (x.T @ yd) / M

    if (k % 10 == 0):
        loss, score = evaluate(x_test, y_test, y_test_one, W)
        history = np.vstack((history,
            np.array([k, loss, score])))
#        print("epoch = %d loss = %f score = %f" 
#             % (k, loss, score))

"""### 結果確認"""

# 損失函數值與準確率確認
print('初始狀態: 損失函數:%f 準確率:%f' 
     % (history[0,1], history[0,2]))
print('最終狀態: 損失函數:%f 準確率:%f' 
     % (history[-1,1], history[-1,2]))

# 學習曲線格式 (損失函數)
plt.plot(history[:,0], history[:,1])
plt.grid()
plt.ylim(0,1.2)
plt.xlabel('iter', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.title('iter vs loss', fontsize=14)
plt.show()

# 學習曲線格式 (準確率)
plt.plot(history[:,0], history[:,2])
plt.ylim(0,1)
plt.grid()
plt.xlabel('iter', fontsize=14)
plt.ylabel('accuracy', fontsize=14)
plt.title('iter vs accuracy', fontsize=14)
plt.show()


# 評價
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 驗證資料計算
yp_test_one = pred(x_test, W)
yp_test = np.argmax(yp_test_one, axis=1)

# 計算準確率
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, yp_test)
print('accuracy: %f' % score)


#from sklearn.metrics import confusion_matrix
#print(confusion_matrix(y_test, yp_test))
#print(classification_report(y_test, yp_test))


"""# 輸入資料改為 4 維 """

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# pdf輸出用
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')


# 加入虛擬變數
x_all2 = np.insert(x_org, 0, 1.0, axis=1)

# 分割訓練資料與驗證資料
from sklearn.model_selection import train_test_split

x_train2, x_test2, y_train, y_test, y_train_one, y_test_one = train_test_split(
    x_all2, y_org, y_all_one, train_size=75, 
    test_size=75, random_state=123)
print(x_train2.shape, x_test2.shape, 
    y_train.shape, y_test.shape, 
    y_train_one.shape, y_test_one.shape)

print('輸入資料(x)')
print(x_train2[:5,:])

# 選擇訓練資料
x, yt, x_test  = x_train2, y_train_one, x_test2

# 初始化處理

# 樣本數
M  = x.shape[0]
# 輸入維數（含虛擬變數）
D = x.shape[1]
# 有幾個分類
N = yt.shape[1]

# 迭代運算次數
iters = 10000

# 學習率
alpha = 0.01

# 權重矩陣初始化
W = np.ones((D, N)) 

# 評估結果記錄
history = np.zeros((0, 3))

# 主程式(４ 維版)
for k in range(iters):
    
    # 計算預測值 (9.7.1)　(9.7.2)
    yp = pred(x, W)
    
    # 計算誤差 (9.7.4)
    yd = yp - yt

    # 更新權重 (9.7.5)
    W = W - alpha * (x.T @ yd) / M

    if (k % 10 == 0):
        loss, score = evaluate(x_test, y_test, y_test_one, W)
        history = np.vstack((history, np.array([k, loss, score])))
#        print("epoch = %d loss = %f score = %f" % (k, loss, score))

print(history.shape)

# 損失函數值與準確率的確認
print('初始狀態: 損失函數:%f 準確率:%f' 
     % (history[0,1], history[0,2]))
print('最終狀態: 損失函數:%f 準確率:%f' 
     % (history[-1,1], history[-1,2]))

# 學習曲線格式 (損失函數值)
plt.plot(history[:,0], history[:,1])
plt.ylim(0,1.2)
plt.grid()
plt.xlabel('iter', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.title('iter vs loss', fontsize=14)
plt.show()

# 學習曲線格式 (準確率)
plt.plot(history[:,0], history[:,2])
plt.ylim(0,1)
plt.grid()
plt.xlabel('iter', fontsize=14)
plt.ylabel('accuracy', fontsize=14)
plt.title('iter vs accuracy', fontsize=14)
plt.show()