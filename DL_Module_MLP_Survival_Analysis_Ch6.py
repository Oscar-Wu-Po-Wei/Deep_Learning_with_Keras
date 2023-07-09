# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 22:45:47 2022

@author: oscar.wu
"""

import numpy as np
import pandas as pd

# 載入資料集
df = pd.read_csv("./titanic_data.csv")

# 顯示資料集的形狀
print(df.shape)

# 查看前5筆記錄
print(df.head())
df.head().to_html("titanic_data_head.html")

# 顯示資料集的描述資料
print(df.describe())
df.describe().to_html("titanic_data_describe.html")
# 可以透過count欄位資訊，初步判斷哪些變項有遺漏數值。

# 如果發現資料集某些欄位有遺漏數值，可使用info()進一步檢視各個欄位是否有遺漏數值。
# 顯示資料集欄位名稱(變數種類)、有資料記數與資料型態。
print(df.info())

# 顯示空值位置。
print(df.isnull())

# 顯示各變數空值(加總)數量(筆數)。
print(df.isnull().sum())

# 資料預處理-Pandas
# Step1. 決定何者為特徵資料，並且只保留特徵資料(以進行Feature Extraction)。
# Step2. 處理遺漏值。
# Step3. 分類資料轉換。

# Step1. 刪除非特徵值欄位。
df = df.drop(["name", "ticket", "cabin"], axis=1)                              # axis=1依照欄位，橫向，由左至右。

# Step2. 處理遺失資料。
df[["age"]] = df[["age"]].fillna(value=df[["age"]].mean())                     # 簡寫：df["age"] = df["age"].fillna(df["age"].mean())
df[["fare"]] = df[["fare"]].fillna(value=df[["fare"]].mean())                  # 簡寫：df["fare"] = df["fare"].fillna(df["fare"].mean())
print(df.isnull().sum())
df.to_excel("titanic_fillna.xlsx")

# df[["欄位名稱"]].value_counts() # 選定欄位，以其欄位內容分類，分別計算資料筆數。
df[["embarked"]] = df[["embarked"]].fillna(value=df["embarked"].value_counts().idxmax())
# value_counts()需要輸入Series資料而非DataFrame資料。
print(df["embarked"].value_counts())
print(df["embarked"].value_counts().idxmax())

"""
print("取得單一欄位資料(型別為Series)")
print(df["欄位名稱1"])
 
print("=================================")
 
print("取得單一欄位資料(型別為DataFrame)")
print(df[["欄位名稱1"]])
 
print("=================================")
 
print("取得多欄位資料(型別為DataFrame)")
print(df[["欄位名稱1", "欄位名稱2"]])
"""

# 類別資料轉換為編碼數值。
df["sex"] = df["sex"].map({"male":1, "female":0}).astype(int)

# One-hot編碼方法
# 方法一：map()將文字轉為數值編碼。
# 方法二：使用get_dummies()將欄位依照文字分類，分別編碼(True=1；False=0)。

# embarked欄位的One-hot編碼
# get_dummies()將embarked欄位依欄位值分類編碼。
embarked_one_hot = pd.get_dummies(df["embarked"], prefix="embarked")
df = df.drop("embarked", axis=1)
df =df.join(embarked_one_hot)

df_survived = df.pop("survived")
df["survived"] = df_survived
print(df.head())

mask = np.random.rand(len(df)) < 0.8                                           # 隨機，並非定值。
print(len(df))
df_train = df[mask]
df_test = df[~mask]
print("Train:",df_train.shape)
print("Test:",df_test.shape)

df_train.to_csv("titanic_train.csv", index=False)
df_test.to_csv("titanic_test.csv", index=False)

from keras.models import Sequential
from keras.layers import Dense

seed = 7
np.random.seed(seed)

# Step1. 資料前處理

# 讀入資料(形成DataFrame形式)
df_train = pd.read_csv("./titanic_train.csv")
df_test = pd.read_csv("./titanic_test.csv")

# 使用values屬性取出資料集形成NumPy陣列。
dataset_train = df_train.values
dataset_test = df_test.values

# 分割成特徵(X)與標籤資料(Y)
X_train = dataset_train[:, 0:9]
Y_train = dataset_train[:, 9]
X_test = dataset_test[:, 0:9]
Y_test = dataset_test[:, 9]

# 資料標準化(只需要針對特徵資料就好)
X_train -= X_train.mean(axis=0)
X_train /= X_train.std(axis=0)
X_test -= X_test.mean(axis=0)
X_test /= X_test.std(axis=0)

# Step2. 定義模型 (如何決定神經元數量與神經網路架構)
model = Sequential()
model.add(Dense(11, input_dim=X_train.shape[1], activation="relu"))
model.add(Dense(11, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.summary()

# Step3. 編譯模型(轉換成低階TensorFlow計算圖)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Step4. 訓練模型
history = model.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=10)

# Step5. 評估模型
loss, accuracy = model.evaluate(X_train, Y_train)
print("訓練資料集的準確度 = {:.2f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test)
print("測試資料集準確度 = {:.2f}".format(accuracy))

import matplotlib.pyplot as plt

loss = history.history["loss"]
epochs = range(1, len(loss)+1)
val_loss = history.history["val_loss"]
plt.plot(epochs, loss, "b-", label="Training Loss")
plt.plot(epochs, val_loss, "r--", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# print(history.history)
acc = history.history["accuracy"]
epochs = range(1,len(acc)+1)
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "b-", label="Training accuracy")
plt.plot(epochs, val_acc, "r--", label="Validation accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()