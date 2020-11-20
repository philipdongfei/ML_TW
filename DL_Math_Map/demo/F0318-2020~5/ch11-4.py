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

"""## 11.6 過度配適解決方法 (常規化)"""

# 正確值的資料數, 將 x 軸設定共有 div 的點
div = 8

# 近似多項式的次方最高項為 dim-1 次方
dim = 8

# y = -x*4 + x**3 -3x**2 + 8x -7
p = [-1, 1, -3, 8, -7]

# x 的定義域介於 [-2, 1]
xMin = -2
xMax = 1

"""$ f(x) = -x^4 + x^3 -3x^2 + 8x -7 + N(0,5) $"""

# x : 在 xMin 與 xMan 之間切出 div 個等差數列
x = np.linspace(xMin, xMax, num=div)

# xx : 將 x 再細切出 div*10 個等差數列
xx = np.linspace(xMin, xMax, num=div*10)

# y, yy: 分別將 x, xx 所有的點代入多項式算出值
y = np.polyval(p, x)
yy = np.polyval(p, xx)

# z: 在 y 中加入隨機產生的常態分佈偏差值 (bias)
z = y + 5 * np.random.randn(div)
print(z)
# 用向量表示的函數
def print_fix(x):
    [print('{:.3f}'.format(n)) for n in x]
    
# 顯示多項式係數的函數    
def print_fix_model(m):
    w = m.coef_.tolist()
    w[0] = m.intercept_
    print_fix(w)

# 產生多項式矩陣做為模型的輸入變數

# x**n 的向量函數 (1, x, x^2, ..., x^(dim-1))
def f(x) :
    return [x**i for i in range(dim)]

# X : 由 x 向量產生的多項式 2 維陣列
X = [f(x0) for x0 in x]

# XX : 由 xx 向量產生的多項式 2 維陣列
XX = [f(x0) for x0 in xx]

# linear regression 模型
from sklearn.linear_model import LinearRegression

# 模型建立與訓練
model = LinearRegression().fit(X, z)

# 算出預測值
yy_pred = model.predict(XX)

# Ridge regression 模型 (L2 regualization)
from sklearn.linear_model import Ridge

# 模型建立與訓練
#model2 = Ridge(alpha=5).fit(X, z)
model2 = Ridge(alpha=0.5).fit(X, z)

# 算出預測值
yy_pred2 = model2.predict(XX)

# 圖形表示
plt.figure(figsize=(8,6))
plt.plot(xx, yy, label='polynomial', lw=1, c='k')
plt.scatter(x, z, label='observed', s=50, c='k')
plt.plot(xx, yy_pred, label='linear regression', lw=3, c='k')
plt.plot(xx, yy_pred2, label='L2 regularizer', lw=3, c='b')
plt.xlabel('X')
plt.ylabel('Y')
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.legend(fontsize=14)
plt.show()