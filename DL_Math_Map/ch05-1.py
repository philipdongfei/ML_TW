import numpy as np
import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

# fig05-01
x = np.linspace(-2, 2, 200)
y = 2**x
x1 = np.linspace(-2, 2, 9)
y1 = 2**x1
plt.figure(figsize=(8,8))
plt.plot(x, y, c='b')
plt.xticks(size=18)
plt.yticks(size=18)
plt.grid(which='major',linestyle='-', lw=2)
plt.scatter(x1, y1, s=50, c='k')
plt.xlabel("$x$", fontsize=20)
plt.ylabel("2^x$", fontsize=20)
plt.show()

# fig05-02
x = np.linspace(-2, 2, 200)
y = (1/2)**x
x1 = np.linspace(-2, 2, 9)
y1 = (1/2)**x1
plt.figure(figsize=(8,8))
plt.plot(x, y, c='b')
plt.scatter(x1, y1, s=50, c='k')
plt.xticks(size=18)
plt.yticks(size=18)
plt.grid(which='major', linestyle='-', lw=2)
plt.xlabel("$x$", fontsize=20)
plt.ylabel(r'$\left(\frac{1}{2}\right)^x$', fontsize=20)
plt.show()

#fig05-05
x = np.linspace(0, 4, 200)
xx = np.linspace(-2, 2, 200)
x0 = np.delete(x, 0)
y0 = np.log2(x0)
x1 = np.linspace(0, 4, 9)
x2 = np.delete(x1, 0)
y2 = np.log2(x2)
plt.figure(figsize=(8,8))
plt.plot(x0, y0, c='b', label='$y=\log_{2}x$', lw=2)
plt.plot(xx, 2**xx, c='k', label='$y=2^x$', lw=2)
plt.plot([-2,4],[-2,4],linestyle='-.',label='$y=x$', lw=2)
plt.plot([-2,4],[0,0],lw=2,c='k')
plt.plot([0,0],[-2,4],lw=2,c='k')
plt.scatter(x2, y2, s=50, c='b')
plt.scatter(y2, x2, s=30, c='k')
plt.xticks(size=20)
plt.yticks(size=20)
plt.ylim(-2,4)
plt.grid(which='major', linestyle='-', lw=2)
plt.xlabel("$x$", fontsize=18)
plt.ylabel("$y$", fontsize=18)
plt.legend(fontsize=18)
plt.show()




