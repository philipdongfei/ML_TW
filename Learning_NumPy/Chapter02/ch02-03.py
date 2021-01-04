import numpy as np

print("all():")

a = np.array([
    [1,1,1],
    [1,0,1],
    [1,0,1],
])
print(np.all(a))

b = np.ones((3,3))
print(np.all(b))
print(np.all(a<2))
print(np.all(b%3<2))
print(np.all(a, axis=0))
print(np.all(a, axis=1))

print(np.all(a, axis=0, keepdims=True))

print(a.all())
print(b.all())
print(a.all(axis=1))
print((a<2).all())
print(a.all(keepdims=True))

print("any():")
a = np.random.randint(10,size=(2,3))
print("a:")
print(a)
print(np.any(a==9))
print(np.any(a==5))
print(np.any(a%2==0, axis=0))
print(np.any(a%2==1, axis=1))
print(np.any(a%2==1, axis=1, keepdims=True))
print(np.any(a>2,keepdims=True))
print((a%5==0).any())
print((a>3).any())

b = np.random.randint(10, size=(2,3))
print("b:")
print(b)
print((a==b).any(axis=1))
print((a==b).any(axis=1, keepdims=True))




