import numpy as np

a =np.array([0,1,2])
b = np.array([4,0])
aa, bb = np.meshgrid(a,b)

print(aa)
print(bb)

aa2, bb2 = np.meshgrid(a, b, indexing='xy')
print(aa2)
print(bb2)

aa3, bb3 = np.meshgrid(a, b, indexing='ij')
print(aa3)
print(bb3)

av, bv = np.meshgrid(a, b, sparse=True)
print(av)
print(bv)

