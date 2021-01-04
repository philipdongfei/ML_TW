import numpy as np

a = np.random.randint(0, 100, size=20)

print(a)
print(np.sort(a))

a = np.array([1,3,2])
print(np.argsort(a))

b = np.random.randint(0,100,size=20).reshape(4,5)
print(b)
print(np.sort(b))
print(np.argsort(b))
print(np.sort(b,axis=0))
print(np.argsort(b, axis=0))

c = np.random.randint(0,100,size=(2,4,5))
print(c)
print(np.sort(c,axis=0))
print(np.argsort(c,axis=0))

values = [('Alice',25,9.7),('Bob',12,7.6),('Catherine',1,8.6),('David',10,7.6)]
dt = [('name', 'S10'),('ID',int),('score',float)]
a = np.array(values, dtype=dt)
print(a)
print(np.sort(a, order='score'))
print(np.argsort(a, order='score'))
print(np.sort(a, order=['score','ID']))

a = np.random.randint(0,100,20)
print(a)
print(np.sort(a))
print(a)
a.sort()
print(a)



