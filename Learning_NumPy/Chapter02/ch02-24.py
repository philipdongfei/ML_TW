import numpy as np

print(np.random.rand())

print(np.random.randint(10))

print(np.random.rand(2,3))

print(np.random.randint(10,size=(2,3)))

print(np.random.randint(5,10,size=10))

print((10-5)*np.random.rand(10) + 5)

np.random.seed(seed=21)
print(np.random.rand())
np.random.seed(21)
print(np.random.rand())
np.random.seed(10)
print(np.random.rand(20))
np.random.seed(23)
print(np.random.rand(20))
np.random.seed(10)
print(np.random.rand(20))
np.random.seed(23)
print(np.random.rand(20))

a = ['Python', 'Ruby', 'Java', 'JavaScipt', 'PHP']

print(np.random.choice(a, 3))
print(np.random.choice(a, 5, replace=False))
print(np.random.choice(a, 20, p = [0.8,0.05,0.05,0.05,0.05]))
print(np.random.choice(5,10))
a = np.arange(10)
print(a)
np.random.shuffle(a)
print(a)





