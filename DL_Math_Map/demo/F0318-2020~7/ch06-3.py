# -*- coding: utf-8 -*-
"""
by makaishi2
"""

import matplotlib.pylab as plt
import numpy as np
x = np.arange(-6, 6, 0.1)

# sigmoid function
sg = 1 / (1 + np.exp(-x))

# sigmoid derivatives
sig = sg*(1-sg)

# normal distribution, mu=0, sigma=1.6
std = np.exp(-x**2 / (2*1.6*1.6)) / (1.6 * np.sqrt(2 * np.pi))

plt.plot(x, sig)
plt.plot(x, std)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()