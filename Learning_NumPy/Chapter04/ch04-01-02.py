import numpy as np

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

b = np.random.randint(10,size=(2,5))
print(b)
c = min_max(b)
print(c)
d = min_max(b,axis=1)
print(d)

