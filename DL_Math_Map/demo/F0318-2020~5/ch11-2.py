# -*- coding: utf-8 -*-
"""
@author: makaishi
"""

import numpy as np

# 以 e 為底的指數函數定義
def f(x):
   return np.exp(x)

# 定義 h 的值
h = 0.001

# 計算 f(0) 微分的近似值
# f'(0)=f(0) 應接近 1
diff = (f(0 + h) - f(0 - h))/(2 * h)

# 結果確認
print(diff)
