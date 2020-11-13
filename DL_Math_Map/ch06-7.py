import numpy as np
import matplotlib.pyplot as plt

def L(p, n, k):
    return ((p ** k) * ((1-p) ** (n-k)))

x = np.linspace(0, 1, 1000)
y = L(x, 5, 2)
x0 = np.asarray([0.4, 0.4])
y0 = np.asarray([0, L(0.4, 5, 2)])

plt.figure(figsize=(6,6))
plt.plot(x, y, c='b', lw=3)
plt.plot(x0, y0, linestyle='dashed', c='k', lw=3)
plt.xticks(size=16)
plt.yticks(size=16)
plt.grid(lw=2)
plt.xlabel("p", fontsize=16)
plt.ylabel("L(p)", fontsize=16)
plt.show()

