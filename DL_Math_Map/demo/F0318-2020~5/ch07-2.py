# -*- coding: utf-8 -*-
"""
by makaishi2
"""

import numpy as np

# 向量與向量的內積
# w = (1, 2)
w = np.array([1, 2])
print(w)
print(w.shape)

# x = (3, 4)
x = np.array([3, 4])
print(x)
print(x.shape)

# （3.7.2）式的內積實作範例
# y = 1*3 + 2*4 = 11
y = x @ w
print(y)


# 矩陣與向量的內積
# Ｘ 為 3 列 2 行的矩陣
X = np.array([[1,2],[3,4],[5,6]])
print(X)
print(X.shape)

Y = X @ w
print(Y)
print(Y.shape)

# 資料矩陣, 向量的內積
# 轉置矩陣
XT = X . T
print(X)
print(XT)

yd = np.array([1, 2, 3])
print(yd)

# 梯度計算
grad = XT @ yd
print(grad)

loss = np.mean(yd ** 2) / 2
print(loss)
