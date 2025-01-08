import pickle


# 读取.pkl文件的函数
def load_data(filepath):
    with open(filepath, 'rb') as pkl_file:
        return pickle.load(pkl_file)


# tau 和 delta 的值列表
taus = [0.2, 0.4, 0.6, 0.8, 1.0]
deltas = [50, 100, 300, 500]

# 存储结果
max_values = []  # 格式: (max_y, corresponding_x, tau, delta)

# 遍历 tau 和 delta
for tau in taus:
    for delta in deltas:
        # 构建文件路径
        filepath = f'BA_500spin/eco_tau{tau}_delta{delta}/network/test_scores.pkl'

        try:
            # 加载数据
            data = load_data(filepath)
            # 解包数据
            x, y = zip(*data)
            # 找到 y 的最大值及其对应的 x
            max_y = max(y)
            corresponding_x = x[y.index(max_y)]
            # 记录最大值和参数
            max_values.append((max_y, corresponding_x, tau, delta))
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

# 打印结果
print("Maximum y values with corresponding parameters:")
for max_y, corresponding_x, tau, delta in max_values:
    print(f"tau={tau}, delta={delta} -> max_y={max_y} at x={corresponding_x}")
