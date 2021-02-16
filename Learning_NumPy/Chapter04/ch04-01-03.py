import numpy as np

np.random.seed(seed=7)

x = np.random.rand(20)*8 - 4
print(x)
y = np.sin(x) + np.random.rand(20)*0.2
omega = np.polyfit(x, y, 5)
print(omega)
f = np.poly1d(omega)

import matplotlib.pyplot as plt
plt.xlabel('x')
plt.ylabel('y')
plt.title('using polyfitfunction')
plt.grid()
plt.scatter(x, y, marker='x', c='red')
xx = np.linspace(-4, 4, 100)
plt.plot(xx, f(xx), color='green')
plt.show()



