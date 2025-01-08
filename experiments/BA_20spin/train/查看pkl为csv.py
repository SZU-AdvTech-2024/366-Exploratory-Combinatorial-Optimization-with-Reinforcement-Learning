import pandas as pd

# 步骤 2: 加载 .pkl 文件
data = pd.read_pickle('BA_500spin//eco_tau0.6_delta300//data//results_BA_500spin_m4_100graphs.pkl')

# 步骤 3: 如果 'data' 不是 DataFrame，请根据实际情况进行转换
# 假设 'data' 是一个字典列表或类似的结构，可以直接转换为 DataFrame
if not isinstance(data, pd.DataFrame):
    data = pd.DataFrame(data)

# 步骤 4: 保存为 .csv 文件
data.to_csv('BA_500spin//eco_tau0.6_delta300//data//results.csv', index=False)  # 设置 index=False 避免保存行索引