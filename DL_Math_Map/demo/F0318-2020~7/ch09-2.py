# -*- coding: utf-8 -*-
"""
by makaishi2
"""

import numpy as np
x = np.array([[1,2,3],[4,5,6]])

# 輸出 x 矩陣內容
print(x)

#輸出第 0 軸加總
y = x.sum(axis=0)
print(y)

#輸出第 1 軸加總
z = x.sum(axis=1)
print(z)
