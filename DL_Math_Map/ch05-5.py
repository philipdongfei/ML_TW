import numpy as np
import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

xx = np.linspace(-6, 6, 500)
yy = 1 / (np.exp(-xx) + 1)

#fig05-10
plt.figure(figsize=(8,6))
plt.ylim(0.0, 1.0)
plt.xlim(-6, 6)
plt.xlabel('x', fontsize=18)
plt.ylabel('y', fontsize=18)
plt.xticks(size=18)
plt.yticks(size=18)
plt.grid(lw=2)
plt.plot(xx, yy, c='b', lw=2, label=r'$\frac{1}{1+\exp{(-x)}}$')
plt.legend(fontsize=20)
plt.show()

plt.figure(figsize=(8, 6))
plt.ylim(-3, 3)
plt.xlim(-3, 3)
plt.xticks(np.linspace(-3,3,13))
plt.yticks(np.linspace(-3,3,13))
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.xticks(size=16)
plt.yticks(size=16)
plt.grid()
plt.plot(xx, yy, c='b', label=r'$\dfrac{1}{1+\exp{(-x)}}$',lw=1)
plt.plot(xx, xx, c='k', label=r'$ = x$', lw=1)
plt.plot([-3,3], [0,0], c='k')
plt.plot([0,0], [-3,3], c='k')
plt.plot([-3,3],[1,1], linestyle='-.', c='r')
plt.legend(fontsize=18)
plt.show()


