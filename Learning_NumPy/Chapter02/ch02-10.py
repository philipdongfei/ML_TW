import numpy as np

print(np.zeros(10))
print(np.zeros(10, dtype=int))
print(np.zeros((3,4)))
a = np.zeros((3,4))

b = np.zeros(a.shape)
print(b)
b = np.zeros_like(a)
print(b)



