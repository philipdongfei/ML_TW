# -*- coding: utf-8 -*-
"""
by makaishi2
"""

import numpy as np
from scipy import integrate
def normal(x):
    return np.exp( -((x-500)**2) / 500) / np.sqrt(500*np.pi)
area=integrate.quad(normal, 0, 480)
print(area)

