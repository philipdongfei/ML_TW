import numpy as np

def zscore(x, axis=None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

a = np.random.randint(10,size=(2,5))
print(a)
print(zscore(a))
print(zscore(a,axis=1))
