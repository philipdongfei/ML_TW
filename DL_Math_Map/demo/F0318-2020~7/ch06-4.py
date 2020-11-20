# -*- coding: utf-8 -*-
"""
by makaishi2
"""

import matplotlib.pylab as plt
import numpy as np

# p 的範圍由 0.0 到 1,0, 間隔 0.01
p = np.arange(0.0, 1.0, 0.01)

# 計算概似函數 y 軸的值
L = np.power(p, 2) * np.power((1-p), 3)

plt.grid(True)
plt.plot(p, L)

plt.xlabel('p')
plt.ylabel('L(p)')
plt.show()