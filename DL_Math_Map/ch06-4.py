import matplotlib.pylab as plt
import numpy as np

# range of p is between 0.0 and 1.0
p = np.arange(0.0, 1.0, 0.01)

# compute y coordinate, L(ikelihood)
L = np.power(p, 2) * np.power((1-p), 3)

plt.grid(True)
plt.plot(p, L)
plt.xlabel('p')
plt.ylabel('L(p)')
plt.show()

