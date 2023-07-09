# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 12:21:36 2022

@author: oscar.wu
"""

# 載入模組與套件
import numpy as np
import pandas as pd
from keras.models import Sequential                                            # Sequential模型
from keras.layers import Dense                                                 # Dense全連接層
from tensorflow.keras.utils import to_categorical                                         # One-hot編碼

# 載入資料集
df = pd.read_csv("./iris.csv")
print(df.shape)

np.random.seed(7)                                                              # 指定亂數種子為7

