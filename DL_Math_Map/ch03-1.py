import numpy as np
import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

def sin(x):
    return (np.sin(x * np.pi / 180.0))

def cos(x):
    return (np.cos(x * np.pi / 180.0))
x = np.linspace(-180.0, 720, 500)

#sin
fig = plt.figure(figsize=(10,3))
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.xlim(-180.0, 720.0)
plt.xticks(np.arange(-180, 810, 90))
plt.ylim(-1.2, 1.2)
plt.grid(lw=2)
plt.plot(x, sin(x), c='b')
plt.plot([-180,721],[0,0], color='black')
plt.plot([0,0],[-1.5,1.5], color='black')
plt.show()

#cos
fig = plt.figure(figsize=(10,3))
plt.xlabel('x',fontsize=14)
plt.ylabel('y',fontsize=14)
plt.xlim(-180.0, 720.0)
plt.xticks(np.arange(-180, 810, 90))
plt.ylim(-1.2, 1.2)
plt.grid(lw=2)
plt.plot(x, cos(x), c='b')
plt.plot([-180,720],[0,0], color='black')
plt.plot([0,0],[-1.5,1.5], color='black')
plt.show()

