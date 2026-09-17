import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=F.relu, use_bias=False):
        super().__init__()
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim)) if use_bias else None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, src_node_features, neighbor_node_features, neighbor_mask):
        neighbor_mask = neighbor_mask.to(dtype=neighbor_node_features.dtype)
        neighbor_sum = (neighbor_node_features * neighbor_mask.unsqueeze(-1)).sum(dim=1)
        neighbor_count = neighbor_mask.sum(dim=1, keepdim=True)
        mean_features = (src_node_features + neighbor_sum) / (neighbor_count + 1.0)
        hidden = torch.matmul(mean_features, self.weight)
        if self.bias is not None:
            hidden = hidden + self.bias
        return self.activation(hidden) if self.activation is not None else hidden


class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_neighbors_list):
        super().__init__()
        self.num_neighbors_list = [int(v) for v in num_neighbors_list]
        self.num_layers = len(self.num_neighbors_list)
        hidden_dim = [int(v) for v in hidden_dim]
        if len(hidden_dim) != self.num_layers:
            raise ValueError('hidden_dim and num_neighbors_list must have the same length')
        dims = [int(input_dim)] + hidden_dim
        self.gcn = nn.ModuleList([SageGCN(dims[i], dims[i + 1], activation=None if i == self.num_layers - 1 else F.relu) for i in range(self.num_layers)])

    def forward(self, node_features_list, neighbor_masks):
        hidden = node_features_list
        for layer in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[layer]
            for hop in range(self.num_layers - layer):
                src = hidden[hop]
                fanout = self.num_neighbors_list[hop]
                neighbors = hidden[hop + 1].view(src.shape[0], fanout, -1)
                mask = neighbor_masks[hop].view(src.shape[0], fanout)
                next_hidden.append(gcn(src, neighbors, mask))
            hidden = next_hidden
        return hidden[0]
