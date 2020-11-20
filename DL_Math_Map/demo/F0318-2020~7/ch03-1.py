# -*- coding: utf-8 -*-
"""ch03-vector.ipynb
by makaishi2

# 3章　向量・矩陣
"""

# 必要 library 宣告
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# PDF輸出用
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

# 三角函數的定義 (可超過 360 度)
def sin(x):
    return(np.sin(x * np.pi / 180.))
def cos(x):
    return(np.cos(x * np.pi / 180.))
x = np.linspace(-180.0, 720, 500)

# 正弦函數(sin)的圖形
fig = plt.figure(figsize=(10,3))
plt.xlabel('x',fontsize=14)
plt.ylabel('y',fontsize=14)
plt.xlim(-180.0, 720.0)
plt.xticks(np.arange(-180, 810, 90))
plt.ylim(-1.2, 1.2)
plt.grid(lw=2)
plt.plot(x, sin(x), c='b')
plt.plot([-180,721],[0,0], color='black')
plt.plot([0,0],[-1.5,1.5],color='black')
plt.show()

# 餘弦函數(cos)的圖形
fig = plt.figure(figsize=(10,3))
plt.xlabel('x',fontsize=14)
plt.ylabel('y',fontsize=14)
plt.xlim(-180.0, 720.0)
plt.xticks(np.arange(-180, 810, 90))
plt.ylim(-1.2, 1.2)
plt.grid(lw=2)
plt.plot(x, cos(x), c='b')
plt.plot([-180,720],[0,0], color='black')
plt.plot([0,0],[-1.5,1.5],color='black')
plt.show()