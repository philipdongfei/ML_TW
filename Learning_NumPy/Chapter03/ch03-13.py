import numpy as np

A = np.array([[1,1,4,0,1],
              [0,3,1,3,2],
              [1,3,0,0,1],
              [2,4,3,1,1]])
print(np.linalg.matrix_rank(A))

B = np.array([
    [1,2,3,0],
    [2,4,6,0],
    [1,0,1,2],
    [1,0,0,3]])
print(np.linalg.matrix_rank(B))

