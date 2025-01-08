import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设 node_embeddings 是所有节点的嵌入
node_embeddings = torch.tensor([
    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # 第一个样本的节点嵌入
    [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]  # 第二个样本的节点嵌入
])

# 假设 n_features 为 2
n_features = 2

# 模拟 layer_pooled 的输出
h_pooled = torch.tensor([
    [1.0, 2.0],  # 第一个样本的全局特征
    [3.0, 4.0]   # 第二个样本的全局特征
])

# 生成 f_pooled
f_pooled = h_pooled.repeat(1, 1, node_embeddings.shape[1]).view(node_embeddings.shape)

print("f_pooled shape:", f_pooled.shape)
print("f_pooled:\n", f_pooled)