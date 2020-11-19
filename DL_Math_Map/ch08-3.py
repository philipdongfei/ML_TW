import numpy as np
import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

from sklearn.linear_model import LogisticRegression
from sklearn import svm

model_lr = LogisticRegression(solver='liblinear')
model_svm = svm.SVC(kernel='linear')

model_lr.fit(x, yt)
model_svm.fit(x, yt)

