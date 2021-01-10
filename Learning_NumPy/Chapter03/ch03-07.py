import numpy as np

a = np.array([[10,5,2,4,9,3,2],
              [10,2,8,3,7,4,1]])
print(np.cov(a))
c = np.array([3,2,1,5,7,2,1])
print(np.cov(a,c))
print(np.cov(a,bias=False))
print(np.cov(a, bias=True))
print(np.cov(a,ddof=None))
print(np.cov(a,ddof=1))
print(np.cov(a,ddof=0))
print(a)
fweights = np.array([1,2,2,1,1,1,1])
print(np.cov(a, fweights=fweights))
aweights = np.array([0.1,0.2,0.2,0.2,0.1,0.1,0.1])
print(np.cov(a, aweights=None))
print(np.cov(a, aweights=aweights))




