import numpy as np

print(np.empty(10))

print(np.empty((2,3)))

%timeit np.zeros(10000)

%timeit np.ones(10000)

%timeit np.empty(10000)

