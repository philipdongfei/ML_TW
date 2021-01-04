import numpy as np
import matplotlib.pyplot as plt
np.random.seed(7)

x = np.random.rand(20)*8-4

y = np.sin(x) + np.random.rand(20)*0.2


omega = np.polyfit(x,y,1) #測試各次方

f=np.poly1d(omega)


plt.xlabel('X')
plt.ylabel('Y')
plt.title('using polyfitfunction')
plt.grid()
plt.scatter(x, y, marker='x', c='red')

xx = np.linspace(-4, 4, 100)

plt.plot(xx, f(xx), color='green')

plt.show()


