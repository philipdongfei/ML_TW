import matplotlib.pylab as plt
import numpy as np

# range of p is between 0.0 and 1.0
p = np.arange(0.0001, 0.9999, 0.01)

# compute y coordinate, L(ikelihood)
LL = 2 * np.log(p) + 3 * np.log(1-p)

plt.grid(True)
plt.plot(p, LL)
plt.xlabel('p')
plt.ylabel('LL(p)')
plt.show()

