# -*- coding: utf-8 -*-
"""ch02-diff.ipynb
# 2章　微分積分
by makaishi2
## 2.1 函數
"""

# 必要 library 宣告
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# PDF輸出用
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

def f(x):
    return x**2 +1

f(1)

f(2)

"""### 圖2-2 畫出(x, f(x))點, 以及 y=f(x)圖形"""

x = np.linspace(-3, 3, 601)
y = f(x)

x1 = np.linspace(-3, 3, 7)
y1 = f(x1)
plt.figure(figsize=(6,6))
plt.ylim(-2,10)
plt.plot([-3,3],[0,0],c='k')
plt.plot([0,0],[-2,10],c='k')
plt.scatter(x1,y1,c='k',s=50)
plt.grid()
plt.xlabel('x',fontsize=14)
plt.ylabel('y',fontsize=14)
plt.show()

x2 = np.linspace(-3, 3, 31)
y2 = f(x2)
plt.figure(figsize=(6,6))
plt.ylim(-2,10)
plt.plot([-3,3],[0,0],c='k')
plt.plot([0,0],[-2,10],c='k')
plt.scatter(x2,y2,c='k',s=50)
plt.grid()
plt.xlabel('x',fontsize=14)
plt.ylabel('y',fontsize=14)
plt.show()

plt.figure(figsize=(6,6))
plt.plot(x,y,c='k')
plt.ylim(-2,10)
plt.plot([-3,3],[0,0],c='k')
plt.plot([0,0],[-2,10],c='k')
plt.scatter([1,2],[2,5],c='k',s=50)
plt.grid()
plt.xlabel('x',fontsize=14)
plt.ylabel('y',fontsize=14)
plt.show()

"""## 2.2 合成函數, 反函數

### 圖2.6 反函數圖形
"""

def f(x):
    return(x**2 + 1)
def g(x):
    return(np.sqrt(x - 1))

xx1 = np.linspace(0.0, 4.0, 200)
xx2 = np.linspace(1.0, 4.0, 200)
yy1 = f(xx1)
yy2 = g(xx2)

plt.figure(figsize=(6,6))
plt.xlabel('$x$',fontsize=14)
plt.ylabel('$y$',fontsize=14)
plt.ylim(-2.0, 4.0)
plt.xlim(-2.0, 4.0)
plt.grid()
plt.plot(xx1,yy1, linestyle='-', c='k', label='$y=x^2+1$')
plt.plot(xx2,yy2, linestyle='-.', c='k', label='$y=\sqrt{x-1}$')
plt.plot([-2,4],[-2,4], color='black')
plt.plot([-2,4],[0,0], color='black')
plt.plot([0,0],[-2,4],color='black')
plt.legend(fontsize=14)
plt.show()

"""## 2.3 微分與極限

### 圖2-7 函數圖形以及局部放大的圖形
"""

from matplotlib import pyplot as plt
import numpy as np

def f(x):
    return(x**3 - x)

delta = 2.0
x = np.linspace(0.5-delta, 0.5+delta, 200)
y = f(x)
fig = plt.figure(figsize=(6,6))
plt.ylim(-3.0/8.0-delta, -3.0/8.0+delta)
plt.xlim(0.5-delta, 0.5+delta)
plt.plot(x, y, 'b-', lw=1, c='k')
plt.scatter([0.5], [-3.0/8.0])
plt.xlabel('x',fontsize=14)
plt.ylabel('y',fontsize=14)
plt.grid()
plt.title('delta = %.4f' % delta, fontsize=14)
plt.show()

delta = 0.2
x = np.linspace(0.5-delta, 0.5+delta, 200)
y = f(x)
fig = plt.figure(figsize=(6,6))
plt.ylim(-3.0/8.0-delta, -3.0/8.0+delta)
plt.xlim(0.5-delta, 0.5+delta)
plt.plot(x, y, 'b-', lw=1, c='k')
plt.scatter([0.5], [-3.0/8.0])
plt.xlabel('x',fontsize=14)
plt.ylabel('y',fontsize=14)
plt.grid()
plt.title('delta = %.4f' % delta, fontsize=14)
plt.show()

delta = 0.01
x = np.linspace(0.5-delta, 0.5+delta, 200)
y = f(x)
fig = plt.figure(figsize=(6,6))
plt.ylim(-3.0/8.0-delta, -3.0/8.0+delta)
plt.xlim(0.5-delta, 0.5+delta)
plt.plot(x, y, 'b-', lw=1, c='k')
plt.scatter(0.5, -3.0/8.0)
plt.xlabel('x',fontsize=14)
plt.ylabel('y',fontsize=14)
plt.grid()
plt.title('delta = %.4f' % delta, fontsize=14)
plt.show()

"""### 圖2-8 函數圖形以及其上兩點連成的直線"""

delta = 2.0
x = np.linspace(0.5-delta, 0.5+delta, 200)
x1 = 0.6
x2 = 1.0
y = f(x)
fig = plt.figure(figsize=(6,6))
plt.ylim(-1, 0.5)
plt.xlim(0, 1.5)
plt.plot(x, y, 'b-', lw=1, c='k')
plt.scatter([x1, x2], [f(x1), f(x2)], c='k', lw=1)
plt.plot([x1, x2], [f(x1), f(x2)], c='k', lw=1)
plt.plot([x1, x2, x2], [f(x1), f(x1), f(x2)], c='k', lw=1)
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
plt.tick_params(color='white')
plt.show()

"""### 圖2-10 切線方程式"""

def f(x):
    return(x**2 - 4*x)
def g(x):
    return(-2*x -1)

x = np.linspace(-2, 6, 500)
fig = plt.figure(figsize=(6,6))
plt.scatter([1],[-3],c='k')
plt.plot(x, f(x), 'b-', lw=1, c='k')
plt.plot(x, g(x), 'b-', lw=1, c='b')
plt.plot([x.min(), x.max()], [0, 0], lw=2, c='k')
plt.plot([0, 0], [g(x).min(), f(x).max()], lw=2, c='k')
plt.grid(lw=2)
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
plt.tick_params(color='white')
plt.show()

"""## 2.4 極大值, 極小值

### 圖2-11 ｙ= x^3-3x 的極大極小
"""

def f1(x):
    return(x**3 - 3*x)

x = np.linspace(-3, 3, 500)
y = f1(x)
fig = plt.figure(figsize=(6,6))
plt.ylim(-4, 4)
plt.xlim(-3, 3)
plt.plot(x, y, 'b-', lw=1, c='k')
plt.plot([0,0],[-4,4],c='k')
plt.plot([-3,3],[0,0],c='k')
plt.grid()
plt.show()

"""### 圖2-12 沒有極大與極小值的例子(y=x^3)"""

def f2(x):
    return(x**3)

x = np.linspace(-3, 3, 500)
y = f2(x)
fig = plt.figure(figsize=(6,6))
plt.ylim(-4, 4)
plt.xlim(-3, 3)
plt.plot(x, y, 'b-', lw=1, c='k')
plt.plot([0,0],[-4,4],c='k')
plt.plot([-3,3],[0,0],c='k')
plt.grid()
plt.show()

"""## 2.7 合成函數的微分

### 圖2-14 反函數的微分
"""

#反函數的微分
def f(x):
    return(x**2 + 1)
def g(x):
    return(np.sqrt(x - 1))

xx1 = np.linspace(0.0, 4.0, 200)
xx2 = np.linspace(1.0, 4.0, 200)
yy1 = f(xx1)
yy2 = g(xx2)

plt.figure(figsize=(6,6))
plt.xlabel('$x$',fontsize=14)
plt.ylabel('$y$',fontsize=14)
plt.ylim(-2.0, 4.0)
plt.xlim(-2.0, 4.0)
plt.grid()
plt.plot(xx1,yy1, linestyle='-', color='blue')
plt.plot(xx2,yy2, linestyle='-', color='blue')
plt.plot([-2,4],[-2,4], color='black')
plt.plot([-2,4],[0,0], color='black')
plt.plot([0,0],[-2,4],color='black')
plt.show()

"""## 2.9 積分

### 圖2-15 面積函數F(x)與f(x)的關係
"""

def f(x) :
    return x**2 + 1
xx = np.linspace(-4.0, 4.0, 200)
yy = f(xx)

plt.figure(figsize=(6,6))
plt.xlim(-2,2)
plt.ylim(-1,4)
plt.plot(xx, yy)
plt.plot([-2,2],[0,0],c='k',lw=1)
plt.plot([0,0],[-1,4],c='k',lw=1)
plt.plot([0,0],[0,f(0)],c='b')
plt.plot([1,1],[0,f(1)],c='b')
plt.plot([1.5,1.5],[0,f(1.5)],c='b')
plt.plot([1,1.5],[f(1),f(1)],c='b')
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
plt.tick_params(color='white')
plt.show()

"""### 圖2-16 面積與定積分"""

plt.figure(figsize=(6,6))
plt.xlim(-2,2)
plt.ylim(-1,4)
plt.plot(xx, yy)
plt.plot([-2,2],[0,0],c='k',lw=1)
plt.plot([0,0],[-1,4],c='k',lw=1)
plt.plot([0,0],[0,f(0)],c='b')
plt.plot([1,1],[0,f(1)],c='b')
plt.plot([1.5,1.5],[0,f(1.5)],c='b')
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
plt.tick_params(color='white')
plt.show()

"""### 圖2-17 積分與面積的關係"""

def f(x) :
    return x**2 + 1
x = np.linspace(-1.0, 2.0, 200)
y = f(x)
N = 10
xx = np.linspace(0.5, 1.5, N+1)
yy = f(xx)

print(xx)

plt.figure(figsize=(6,6))
plt.xlim(-1,2)
plt.ylim(-1,4)
plt.plot(x, y)
plt.plot([-1,2],[0,0],c='k',lw=2)
plt.plot([0,0],[-1,4],c='k',lw=2)
plt.plot([0.5,0.5],[0,f(0.5)],c='b')
plt.plot([1.5,1.5],[0,f(1.5)],c='b')
plt.bar(xx[:-1], yy[:-1], align='edge', width=1/N*0.9)
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
plt.tick_params(color='white')
plt.grid()
plt.show()