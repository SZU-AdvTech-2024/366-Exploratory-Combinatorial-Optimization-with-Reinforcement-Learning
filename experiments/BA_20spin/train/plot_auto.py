import pickle
import matplotlib.pyplot as plt


# 读取.pkl文件的函数
def load_data(filepath):
    with open(filepath, 'rb') as pkl_file:
        return pickle.load(pkl_file)


# tau 和 delta 的值列表
taus = [0.2, 0.4, 0.6, 0.8, 1.0]
deltas = [50, 100, 300, 500]

# 图表设置
plt.figure(figsize=(10, 6))

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

            # 创建新图
            plt.figure(figsize=(8, 5))
            plt.plot(x, y, marker='o', linestyle='-', label=f'tau={tau}, delta={delta}')

            # 添加标题、标签和图例
            plt.title(f'Trend of Y with respect to X\n(tau={tau}, delta={delta})')
            plt.xlabel('X Value (e.g., Time or Distance)')
            plt.ylabel('Y Value (Measured/Calculated)')
            plt.legend()
            plt.grid(True)

            # 显示单独的图
            plt.show()
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
