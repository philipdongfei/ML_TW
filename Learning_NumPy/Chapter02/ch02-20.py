import numpy as np

print(np.newaxis is None)

x = np.arange(15).reshape(3,5)
print(x)
print(x.shape)

y = x[np.newaxis,:,:]
print(y)
print(y.shape)

z = x[:,np.newaxis,:]
print(z)
print(z.shape)
print(x[:,None,:])

x = x.flatten()
print(x)
print(x.shape)
print(x[:,np.newaxis])
print(x[:,np.newaxis].shape)

x = np.arange(15).reshape(3,5)
print(np.reshape(x,(1,3,5)))
print(np.reshape(x,(3,1,5)))
x = x.flatten()
print(np.reshape(x,(-1,1)))
print(np.reshape(x,(-1,1)).shape)










