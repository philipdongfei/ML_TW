import numpy as np
import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

def f(x):
    return x**2 + 1

x = np.linspace(-3, 3, 601)
y = f(x)

x1 = np.linspace(-3, 3, 7)
y1 = f(x1)
plt.figure(figsize=(6,6))
plt.ylim(-2, 10)
plt.plot([-3, 3], [0,0], c='k')
plt.plot([0, 0], [-2,10], c='k')
plt.scatter(x1,y1,c='k',s=50)
plt.grid()
plt.xlabel('x', fontsize=14)
plt.ylabel('y',fontsize=14)
plt.show()

x2 = np.linspace(-3, 3, 31)
y2 = f(x2)
plt.figure(figsize=(6,6))
plt.ylim(-2, 10)
plt.plot([-3, 3], [0,0], c='k')
plt.plot([0, 0], [-2,10], c='k')
plt.scatter(x2,y2,c='k',s=50)
plt.grid()
plt.xlabel('x', fontsize=14)
plt.ylabel('y',fontsize=14)
plt.show()


plt.figure(figsize=(6,6))
plt.plot(x,y,c='k')
plt.ylim(-2,10)
plt.plot([-3, 3], [0,0], c='k')
plt.plot([0, 0], [-2,10], c='k')
plt.scatter([1,2],[2,5],c='k',s=50)
plt.grid()
plt.xlabel('x', fontsize=14)
plt.ylabel('y',fontsize=14)
plt.show()

