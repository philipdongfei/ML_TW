import numpy as np

print("amax():")
print(np.amax(np.array([1,2,3,2,1])))

arr = np.array([1,2,3,4]).reshape(2,2)
print(arr)
print(np.amax(arr, axis=0))
print(np.amax(arr, axis=1))
print(np.amax(arr,keepdims=True))

b = np.random.randint(30,size=(2,3,5))
print(b)
print(b.max(axis=0))
print(b.max(axis=1))
print(b.max(axis=2))

print("amin():")
a = np.array([
    [1.2,1.3,0.1,1.5],
    [2.1,0.2,0.3,2.0],
    [0.1,0.5,0.5,2.3]
])

print(a)
print(np.amin(a))
print(np.amin(a, axis=0))
print(np.amin(a, axis=1))
print(np.amin(a, axis=0, keepdims=True))
print(np.amin(a, axis=1, keepdims=True))

