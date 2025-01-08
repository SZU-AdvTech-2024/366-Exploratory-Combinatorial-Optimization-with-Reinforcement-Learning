import torch
import torch.nn as nn
import torch.nn.functional as F

class MPNN(nn.Module):
    def __init__(self,
                 n_obs_in=7,
                 n_layers=3,
                 n_features=64,
                 tied_weights=False,
                 tau=0.5,
                 delta=100,
                 n_hid_readout=[],):

        super().__init__()

        self.n_obs_in = n_obs_in
        self.n_layers = n_layers
        self.n_features = n_features
        self.tied_weights = tied_weights
        self.tau=tau
        self.delta=delta
        self.node_init_embedding_layer = nn.Sequential(
            nn.Linear(n_obs_in, n_features, bias=False),
            nn.ReLU()
        )

        self.edge_embedding_layer = EdgeAndNodeEmbeddingLayer(n_obs_in, n_features)
        #创建重要性评分层
        #self.importance_scoring_network = ImportanceScoringNetwork()

        if self.tied_weights:
            self.update_node_embedding_layer = UpdateNodeEmbeddingLayer(n_features)
        else:
            self.update_node_embedding_layer = nn.ModuleList([UpdateNodeEmbeddingLayer(n_features) for _ in range(self.n_layers)])

        self.readout_layer = ReadoutLayer(n_features, n_hid_readout)

    @torch.no_grad()
    def get_normalisation(self, adj):
        norm = torch.sum((adj != 0), dim=1).unsqueeze(-1)
        norm[norm == 0] = 1
        return norm.float()
        
    def forward(self, obs):
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        obs.transpose_(-1, -2)

        # Calculate features to be used in the MPNN
        node_features = obs[:, :, 0:self.n_obs_in]

        # Get graph adj matrix.
        adj = obs[:, :, self.n_obs_in:]
        # adj_conns = (adj != 0).type(torch.FloatTensor).to(adj.device)



        # ########################
        # node_degrees = torch.sum(adj , dim=-1).float()  # (batch_size, num_nodes)
        # topk_values, topk_indices = torch.topk(node_degrees, k=10, dim=-1)  # (batch_size, k)
        # sparse_node_degrees = torch.zeros_like(node_degrees)  # (batch_size, num_nodes)
        # for batch_idx in range(obs.shape[0]):
        #     sparse_node_degrees[batch_idx, topk_indices[batch_idx]] = topk_values[batch_idx]
        #
        # importance_scores = sparse_node_degrees.unsqueeze(-1) * sparse_node_degrees.unsqueeze(
        #     -2)  # (batch_size, num_nodes, num_nodes)
        #
        # total_importance = torch.sum(importance_scores, dim=[-2, -1], keepdim=True)  # (batch_size, 1, 1)
        #
        # attention_weights = importance_scores / total_importance  # (batch_size, num_nodes, num_nodes)
        #
        # adj_aggregated=adj+200*attention_weights
        # #######################

        # importance_scores = self.importance_scoring_network(torch.cat([node_features,adj], dim=-1)).squeeze(-1)
        # ###################
        # # 对所有节点的重要性分数进行排序，并选择前k个节点
        # #topk_values, topk_indices = torch.topk(importance_scores, k=10, dim=-1)  # (batch_size, k)
        # # 创建一个全零的张量，用于存储稀疏的节点重要性分数
        # #sparse_importance_scores = torch.zeros_like(importance_scores)  # (batch_size, num_nodes)
        # # 将top-k节点的重要性分数填充到新的张量中
        # #for batch_idx in range(obs.shape[0]):
        # #   sparse_importance_scores[batch_idx, topk_indices[batch_idx]] = topk_values[batch_idx]
        # ###############
        #
        #  # (batch_size, num_nodes)
        # sparse_importance_scores = F.softmax(importance_scores / self.tau, dim=-1)  # 使用带温度参数的softmax函数
        # # 使用稀疏的节点重要性分数计算每条边的重要性分数
        # attention_weights = sparse_importance_scores.unsqueeze(-1) * sparse_importance_scores.unsqueeze(-2)  # (batch_size, num_nodes, num_nodes)
        # # 计算所有节点对的重要性分数总和
        # total_importance = torch.sum(attention_weights, dim=[-2, -1], keepdim=True)  # (batch_size, 1, 1)
        # # 将每个节点对的重要性分数除以总和，得到注意力权重
        # attention_weights = attention_weights / total_importance  # (batch_size, num_nodes, num_nodes)
        # # 按超参数delta进行加权聚合，得到聚合邻接矩阵，之后传入到边嵌入和消息传递过程中
        adj_aggregated = adj #+ self.delta* attention_weights


        init_node_embeddings = self.node_init_embedding_layer(node_features)
        # Initialise embeddings.
        current_node_embeddings = init_node_embeddings
        norm = self.get_normalisation(adj_aggregated)
        edge_embeddings = self.edge_embedding_layer(node_features, adj_aggregated, norm)
        # ##############################


        if self.tied_weights:
            for _ in range(self.n_layers):
                current_node_embeddings = self.update_node_embedding_layer(current_node_embeddings,
                                                                           edge_embeddings,
                                                                           norm,
                                                                           adj_aggregated)
        else:
            for i in range(self.n_layers):
                current_node_embeddings = self.update_node_embedding_layer[i](current_node_embeddings,
                                                                              edge_embeddings,
                                                                              norm,
                                                                              adj_aggregated)

        out = self.readout_layer(current_node_embeddings)
        out = out.squeeze()

        return out

class EdgeAndNodeEmbeddingLayer(nn.Module):

    def __init__(self, n_obs_in, n_features):
        super().__init__()
        self.n_obs_in = n_obs_in
        self.n_features = n_features

        self.edge_embedding_NN = nn.Linear(int(n_obs_in+1), n_features-1, bias=False)
        self.edge_feature_NN = nn.Linear(n_features, n_features, bias=False)

    def forward(self, node_features, adj, norm):
        edge_features = torch.cat([adj.unsqueeze(-1),
                                   node_features.unsqueeze(-2).transpose(-2, -3).repeat(1, adj.shape[-2], 1, 1)],
                                  dim=-1)

        edge_features *= (adj.unsqueeze(-1)!=0).float()

        edge_features_unrolled = torch.reshape(edge_features, (edge_features.shape[0], edge_features.shape[1] * edge_features.shape[1], edge_features.shape[-1]))
        embedded_edges_unrolled = F.relu(self.edge_embedding_NN(edge_features_unrolled))
        embedded_edges_rolled = torch.reshape(embedded_edges_unrolled,
                                              (adj.shape[0], adj.shape[1], adj.shape[1], self.n_features-1))
        embedded_edges = embedded_edges_rolled.sum(dim=2) / norm

        edge_embeddings = F.relu(self.edge_feature_NN(torch.cat([embedded_edges, norm / norm.max()],dim=-1)))

        return edge_embeddings

class UpdateNodeEmbeddingLayer(nn.Module):

    def __init__(self, n_features):
        super().__init__()

        self.message_layer = nn.Linear(2*n_features, n_features, bias=False)
        self.update_layer = nn.Linear(2*n_features, n_features, bias=False)

    def forward(self, current_node_embeddings, edge_embeddings, norm, adj):
        node_embeddings_aggregated = torch.matmul(adj, current_node_embeddings) / norm

        message = F.relu(self.message_layer(torch.cat([node_embeddings_aggregated, edge_embeddings], dim=-1)))
        new_node_embeddings = F.relu(self.update_layer(torch.cat([current_node_embeddings, message], dim=-1)))

        return new_node_embeddings


class ReadoutLayer(nn.Module):

    def __init__(self, n_features, n_hid=[], bias_pool=False, bias_readout=True):

        super().__init__()

        self.layer_pooled = nn.Linear(int(n_features), int(n_features), bias=bias_pool)

        if type(n_hid)!=list:
            n_hid = [n_hid]

        n_hid = [2*n_features] + n_hid + [1]

        self.layers_readout = []
        for n_in, n_out in list(zip(n_hid, n_hid[1:])):
            layer = nn.Linear(n_in, n_out, bias=bias_readout)
            self.layers_readout.append(layer)

        self.layers_readout = nn.ModuleList(self.layers_readout)

    def forward(self, node_embeddings):

        f_local = node_embeddings

        h_pooled = self.layer_pooled(node_embeddings.sum(dim=1) / node_embeddings.shape[1])
        f_pooled = h_pooled.repeat(1, 1, node_embeddings.shape[1]).view(node_embeddings.shape)

        features = F.relu( torch.cat([f_pooled, f_local], dim=-1) )

        for i, layer in enumerate(self.layers_readout):
            features = layer(features)
            if i<len(self.layers_readout)-1:
                features = F.relu(features)
            else:
                out = features

        return out

# class ImportanceScoringNetwork(nn.Module):
#     def __init__(self, input_dim=107, hidden_dim=128, output_dim=1):
#         super(ImportanceScoringNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x