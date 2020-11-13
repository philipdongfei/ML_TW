import numpy as np
import scipy.special as scm
import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

#N = 2
for N in [2,3,4,10]:
    M = 2**N
    X = range(N+1)
    plt.bar(X, [scm.comb(N, i)/M for i in X])
    plt.xticks(X, [str(i) for i in X])
    plt.show()

for N in [100, 1000]:
    M = 2**N
    if N == 100:
        X = range(30,71)
    else:
        X = range(440, 561)
    plt.bar(X, [scm.comb(N,i)/M for i in X])
    plt.show()

