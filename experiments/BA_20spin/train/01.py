import pickle
import csv
import pandas as pd
# 读取.pkl文件
with open('results/500a.pkl', 'rb') as pkl_file:
    data1 = pickle.load(pkl_file)

print(data1)
with open('results/500.pkl', 'rb') as pkl_file:
    data2 = pickle.load(pkl_file)

print(data2)
import matplotlib.pyplot as plt

# 数据
x, y = zip(*data1)
x1, y1 = zip(*data2)
# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(x, y,  linestyle='-', color='b')
plt.plot(x1, y1,  linestyle='-', color='orange')
plt.title('Trend of Y with respect to X')
plt.xlabel('X Value (e.g., Time or Distance)')
plt.ylabel('Y Value (Measured/Calculated)')
plt.grid(True)
plt.show()