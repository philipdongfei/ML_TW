sell_list = [[9984, 8901004], [9987, 2126993], [9983, 1861917], [1333, 873295], [1332, 635953],
             [9989, 528394], [1417, 283236], [1301, 236561], [9974, 193566], [9994, 167334],
             [1419, 157001], [9997, 146083], [9993, 114111], [9977, 105954], [1413, 105007],
             [9991, 98729], [9982, 77952], [9995, 77581], [1352, 73761], [9979, 63957],
             [1379, 63119], [1377, 61844], [1376, 57848], [9990, 56747], [1420, 55504],
             [1414, 53250], [9996, 48505], [9967, 34353], [9976, 33591], [1407, 32753],
             [9978, 29919], [9966, 21387], [1430, 20948], [1418, 19082], [1381, 18802],
             [1429, 18052], [1384, 15982], [9972, 15173], [9980, 13401], [1408, 10599],
             [9969, 8791], [9992, 7281], [9986, 6944], [9973, 5420], [1380, 5399],
             [1383, 4623], [1431, 3765], [1382, 3722], [1401, 2660], [1400, 2355]]
import numpy as np
import matplotlib.pyplot as plt
sell_np = np.array(sell_list)

sell_np = np.array(sell_list)
index = range(sell_np.shape[0])
sc_label = ['' for x in sell_np[:,1]]

# fig05-06
plt.figure(figsize=(10,6))
plt.bar(index, sell_np[:,1], tick_label=sc_label, color='b')
#plt.bar(index,sell_np[:,1], color='b')
plt.grid(lw=2)
plt.xticks([])
plt.yticks(size=16)
plt.show()

# fig05-07
plt.figure(figsize=(10,6))
plt.bar(index, np.log(sell_np[:,1]), tick_label=sc_label, color='b')
plt.ylim(7,17)
plt.grid(lw=2)
plt.xticks([])
plt.yticks(size=16)
plt.show()

