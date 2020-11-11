import numpy as np
import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

def f(x):
    return (x**2 + 1)
def g(x):
    return (np.sqrt(x-1))

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
plt.plot(xx1, yy1, linestyle='-', c='k', label='$y=x^2+1$')
plt.plot(xx2, yy2, linestyle='-.', c='k', label='$y=\sqrt{x-1}$')
plt.plot([-2,4],[-2,4], color='black')
plt.plot([-2,4],[0,0], color='black')
plt.plot([0,0],[-2,4], color='black')
plt.legend(fontsize=14)
plt.show()





