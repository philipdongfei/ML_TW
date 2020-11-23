import numpy as np

#e^x
def f(x):
    return np.exp(x)

h = 0.001

#f'(0)
#f'(0) = f(0) = 1
diff = (f(0+h) - f(0-h))/(2*h)

print(diff)



