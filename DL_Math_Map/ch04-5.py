import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#pdf
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

def f(x):
    return 3*x**4 + 4*x**3 - 12*x**2

def diff(x):
    return x**3 + x**2 - x

xx = np.linspace(-3, 2, 101)
yy = f(xx)

a1 = 0.5
a2 = -0.5

plt.figure(figsize=(6,6))
plt.xticks(color="None")
plt.yticks(color="None")
plt.tick_params(length=0)
plt.plot(xx,yy)
plt.scatter([a1],[f(a1)], s=50, c='k', label='A')
plt.scatter([a2],[f(a2)], s=50, c='b', label='B')
plt.scatter([-2,1],[f(-2), f(1)], s=50, c='k')
plt.quiver([a1,a2],[f(a1),f(a2)],[diff(a1), diff(a2)],[0,0])
plt.show()

