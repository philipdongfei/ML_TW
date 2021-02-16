import numpy as np
import pandas as pd

df = pd.read_csv('iris.data', header=None)

print(df)

x = df.iloc[0:100,[0,1,2,3]].values
print(x)

y = df.iloc[0:100,4].values
print(y)
y = np.where(y=='Iris-setosa',0,1)
print(y)

x_train = np.empty((80,4))
x_test = np.empty((20,4))
y_train = np.empty(80)
y_test = np.empty(20)
x_train[:40],x_train[40:] = x[:40],x[50:90]
x_test[:10],x_test[10:] = x[40:50],x[90:100]
y_train[:40],y_train[40:] = y[:40],y[50:90]
y_test[:10],y_test[10:] = y[40:50],y[90:100]





