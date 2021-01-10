import numpy as np
import timeit

a = np.random.randint(100, size=(2,3,4))
print(a)
print(np.median(a))
print(np.median(a,axis=2))
print(np.median(a,axis=1))
b = a.copy()
print(b)
print(np.median(b, axis=1, overwrite_input=True))
print(np.all(a==b))
print(b)

print(np.median(a, axis=0,keepdims=True))
print(np.median(a, axis=1,keepdims=False))
print(np.median(a, axis=1,keepdims=True))





