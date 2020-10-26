import numpy as np
import scipy.special as scm
import matplotlib.pyplot as plt

def gauss(x, n):
    m = n/2
    return np.exp(-(x-m)**2 / m) / np.sqrt(m * np.pi)

N = 1000
M = 2**N
X = range(440, 561)
plt.bar(X, [scm.comb(N, i)/M for i in X])
plt.plot(X, gauss(np.array(X), N), c='k', linewidth=2)
plt.show()

