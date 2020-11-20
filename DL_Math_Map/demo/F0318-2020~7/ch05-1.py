# -*- coding: utf-8 -*-
"""
by makaishi2
"""

import numpy as np
np.set_printoptions(precision = 10)
x = np.logspace(0, 11, 12, base=0.1, dtype='float64')
y = np.power(1 + x, 1 / x)
for i in range(11):
    print( 'x = %12.10f y = %12.10f' % (x[i], y[i]))
