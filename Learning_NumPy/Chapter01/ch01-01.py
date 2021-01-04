import numpy as np

a = np.array([1,2,3])

print(type(a))

print(a)

print(a*3)

print(a+2)

b = np.array([2,2,0])

print(a+b)

print(a/b)

print(a * b)

print(np.arange(10))

print(np.arange(0,10,2))

print(np.linspace(0, 10, 15))

c = np.array([[1,2,3],[4,5,6]])

print(c)
print(c.shape)

d = np.array([[[1,2,3],[4,5,6],[7,8,9],[10,11,12]],
              [[13,14,14],[16,17,18],[19,20,21],[22,23,24]]])
print(d)
print(d.shape)

print(c.reshape(3, 2))
print(c.reshape(6, 1))

import time

def calculate_time():
    a = np.random.randn(100000)
    b = list(a)
    start_time = time.time()
    for _ in range(1000):
        sum_1 = np.sum(a)
    print("Using NumPy\t %f sec" % (time.time()-start_time))
    start_time = time.time()
    for _ in range(1000):
        sum_2 = sum(b)
    print("Not using NumPy\t %f sec" % (time.time()-start_time))

calculate_time()




