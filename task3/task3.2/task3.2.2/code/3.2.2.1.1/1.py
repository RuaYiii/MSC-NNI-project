#获取去年6月的数据
import pandas as pd
import numpy as np
l = 1
r = 9299168
L = -1
while l <= r:
    m = (l+r)//2
    file = pd.read_csv('./santander-product-recommendation/train_ver2.csv/train_ver2.csv', skiprows = m, nrows = 2)
    if file.values[0][0][6] >= '6': 
        r = m-1
        continue
    if file.values[1][0][6] <= '5':
        l = m+1
        continue
    if file.values[0][0][6] == '5' and file.values[1][0][6] == '6':
        L = m
        break
print(L)
file = pd.read_csv('./santander-product-recommendation/train_ver2.csv/train_ver2.csv', skiprows = L, nrows = 2)
print(file)

l = 1
r = 9299168
L = -1
while l <= r:
    m = (l+r)//2
    file = pd.read_csv('./santander-product-recommendation/train_ver2.csv/train_ver2.csv', skiprows = m, nrows = 2)
    if file.values[0][0][6] >= '7': 
        r = m-1
        continue
    if file.values[1][0][6] <= '6':
        l = m+1
        continue
    if file.values[0][0][6] == '6' and file.values[1][0][6] == '7':
        R = m
        break
print(R)
file = pd.read_csv('./santander-product-recommendation/train_ver2.csv/train_ver2.csv', skiprows = R, nrows = 2)
print(file)

file = pd.read_csv('./santander-product-recommendation/train_ver2.csv/train_ver2.csv', skiprows = L+1, nrows = R-L)
index = pd.read_csv('./santander-product-recommendation/train_ver2.csv/train_ver2.csv', nrows = 0)
file.columns = index.columns.values
file.to_csv("./train.csv",index=0)