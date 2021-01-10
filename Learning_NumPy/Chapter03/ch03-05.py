import numpy as np

a = np.random.rand(10)

print(a)
print(np.std(a))
print()

b = np.random.rand(2,3,4)
print(b)
print(np.std(b,axis=0))

print(np.std(b))
print(np.std(b, ddof=1))

