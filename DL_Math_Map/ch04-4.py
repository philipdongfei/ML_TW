import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#pdf
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

def f(x):
    return x**2

def diff(x):
    return 2*x

x = np.linspace(-2.5, 2.5, 101)
y = f(x)

a1 = 2
b1 = f(a1)
d1 = diff(a1) * 0.1
a2 = 1.5
b2 = f(a2)
d2 = diff(a2) * 0.1
a3 = 1
b3 = f(a3)
d3 = diff(a3) * 0.1
a4 = 0.5
b4 = f(a4)
d4 = diff(a4) * 0.1

plt.figure(figsize=(8,8))
plt.plot(x, y)
plt.arrow(a1, b1, -d1, 0, head_width=0.1, head_length=0.1, color='k')
plt.arrow(a2, b2, -d2, 0, head_width=0.1, head_length=0.1, color='k')
plt.arrow(a3, b3, -d3, 0, head_width=0.1, head_length=0.1, color='k')
plt.arrow(a4, b4, -d4, 0, head_width=0.1, head_length=0.1, color='k')
plt.grid()
plt.show()

