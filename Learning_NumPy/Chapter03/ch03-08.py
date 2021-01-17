import numpy as np

x = np.array([
    [1,2,1,9,10,3,2,6,7],
    [2,1,8,3,7,5,10,7,2]
])
print(np.corrcoef(x))
y = np.array([2,1,1,8,9,4,3,5,7])
print(np.corrcoef(x,y))
print(np.corrcoef(x,rowvar=False))
x_transpose = x.T
print(np.corrcoef(x_transpose, rowvar=False))
print(np.corrcoef(x_transpose, rowvar=True))

