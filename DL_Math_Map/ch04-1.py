import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#pdf
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

def L(u, v):
    return 3 * u**2 + 3 * v**2 - u*v + 7*u - 7*v + 10

def Lu(u, v):
    return 6 * u - v + 7

def Lv(u, v):
    return -u + 6 * v - 7

print(L(-1, 1))
print(L(0, 2))
print(Lu(0,0))
print(Lv(0,0))


