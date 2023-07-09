# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 21:45:13 2022

@author: oscar.wu
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(7) # 指定亂數種子

df = pd.read_csv("./boston_housing.csv")
print(df.shape)
print(df.head())

dataset = df.values #使用values屬性將dataframe資料轉換成NumPy陣列
np.random.shuffle(dataset) # 使用亂數打亂資料，讓每次執行結果可在相同的亂數條件下進行比較、分析與除錯。

# 分割成「特徵資料X」和「標籤資料Y」
X = dataset[:, 0:13]
Y = dataset[:, 13]
# 特徵標準化
X -= X.mean(axis=0)
X /= X.std(axis=0)
# 分割訓練(train)和測試(test)資料集
X_train, Y_train = X[:404], Y[:404]
X_test, Y_test = X[404:], Y[404:]

# 三層神經網路
# 因為是迴歸分析，輸出層無啟動函數(因為線性迴歸分析，不需要經過啟動函數之非線性轉換)。
# MAE(平均絕對誤差)，可真實反映預測值與標籤值誤差的實際情況。
def build_model():
    model = Sequential()
    model.add(Dense(32, input_shape=(X_train.shape[1], ), activation="relu"))  # shape[0]：表示矩陣的行數(column)；shape[1]：表示矩陣的列數(row)。
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])               # Mean Absolute Error = 誤差絕對值的平均，可真實反應預測值與標籤值誤差的實際情形。
    return model

# 使用比較深的四層神經網路    
def build_deep_model():
    model = Sequential()
    model.add(Dense(32, input_shape=(X_train.shape[1], ), activation="relu"))  # 隱藏層一
    model.add(Dense(16, activation="relu"))                                    # 隱藏層二
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam", metrics=["mae"])
    return model
    
"""
Keras 的模型結構分兩種：
Sequential model：一層層順序執行的簡單模型，只有第一層要寫input的規格，其他層的input就是上一層的output。
Functional API model：可以有多個 input 層或 output 層，結構可以有分叉，適合複雜的模型建立。
"""

k = 4                                                                          # 訂定折數，此處為4(=訓練資料集+驗證資料集)，外加一批測試資料集，共5個部分。
nb_val_samples = len(X_train) // k                                             # 每一折的樣本數。
nb_epochs = 80                                                                 # 訓練週期數。
# 記錄每一次評估模型的MSE與MAE數值
mse_scores = []
mae_scores = []

for i in range(k):
    print("Processing Fold #" + str(i))
    # 使用切割運算子取得第K個驗證資料集X_val與Y_val
    X_val = X_train[i*nb_val_samples : (i+1)*nb_val_samples]
    Y_val = Y_train[i*nb_val_samples : (i+1)*nb_val_samples]
    # 使用concatenate()結合剩下的折來建立訓練資料集
    X_train_p = np.concatenate([X_train[:i*nb_val_samples],X_train[(i+1)*nb_val_samples:]], axis=0)
    Y_train_p = np.concatenate([Y_train[:i*nb_val_samples],Y_train[(i+1)*nb_val_samples:]], axis=0)

model = build_model()
# fit()：訓練模型
model.fit(X_train_p, Y_train_p, epochs=nb_epochs,
          batch_size=16, verbose=0)
# evaluate()：評估模型
mse, mae = model.evaluate(X_val, Y_val)
mse_scores.append(mse)
mae_scores.append(mae)

print("MSE_val: ", np.mean(mse_scores))
print("MAE_val: ", np.mean(mae_scores))
mse, mae = model.evaluate(X_test, Y_test)
print("MSE_test: ", mse)
print("MAE_test: ", mae)

# 使用全部的訓練資料來訓練模型
# 使用K-fold交叉驗證分析找出最佳的神經網路結構後，即可使用全部訓練資料來訓練以上建構好的神經網路模型。
model = Sequential()
model.add(Dense(32, input_shape=(X_train.shape[1], ), activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam", metrics=["mae"])
# 使用全部的訓練資料(沒有分割驗證資料)來訓練模型
model.fit(X_train, Y_train, epochs=80, batch_size=16, verbose=0)
mse, mae = model.evaluate(X_test, Y_test)
print("MSE_test: ", mse)
print("MAE_test: ", mae)