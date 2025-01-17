import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import degree
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import to_undirected
from torch_geometric.transforms import FaceToEdge

class FeatureExpander(MessagePassing):
    r"""Expand features.

    Args:
        degree (bool): whether to use degree feature.
        onehot_maxdeg (int): whether to use one_hot degree feature with
            with max degree capped. disableid with 0.
        AK (int): whether to use a^kx feature. disabled with 0.
        centrality (bool): whether to use centrality feature.
        remove_edges (strings): whether to remove edges, partially or totally.
        group_degree (int): group nodes to create super nodes, set 0 to disable.
    """

    def __init__(self, degree=True, onehot_maxdeg=0, AK=1,
                 centrality=False, remove_edges="none", group_degree=0):
        super(FeatureExpander, self).__init__('add', 'source_to_target')
        self.degree = degree
        self.onehot_maxdeg = onehot_maxdeg
        self.AK = AK
        self.centrality = centrality
        self.remove_edges = remove_edges
        self.group_degree = group_degree
        assert remove_edges in ["none", "nonself", "all"], remove_edges

        self.edge_norm_diag = 1e-8  # edge norm is used, and set A diag to it

    def transform(self, data):
        if data.x is None:
          data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)

        deg, deg_onehot = self.compute_degree(data.edge_index, data.num_nodes)
        akx = self.compute_akx(data.num_nodes, data.x, data.edge_index)
        cent = self.compute_centrality(data)
        data.x = torch.cat([data.x, deg, deg_onehot, akx, cent], -1)

        if self.remove_edges != "none":
            if self.remove_edges == "all":
                self_edge = None
            else:  # only keep self edge
                self_edge = torch.tensor(range(data.num_nodes)).view((1, -1))
                self_edge = torch.cat([self_edge, self_edge], 0)
            data.edge_index = self_edge

        # Reduce nodes by degree-based grouping
        if self.group_degree > 0:
            assert self.remove_edges == "all", "remove all edges"
            x_base = data.x
            deg_base = deg.view(-1)
            super_nodes = []
            for k in range(1, self.group_degree + 1):
                eq_idx = deg_base == k
                gt_idx = deg_base > k
                x_to_group = x_base[eq_idx]
                x_base = x_base[gt_idx]
                deg_base = deg_base[gt_idx]
                group_size = torch.zeros([1, 1]) + x_to_group.size(0)
                if x_to_group.size(0) == 0:
                  super_nodes.append(
                      torch.cat([group_size, data.x[:1]*0], -1))
                else:
                  super_nodes.append(
                      torch.cat([group_size,
                                 x_to_group.mean(0, keepdim=True)], -1))
            if x_base.size(0) == 0:
                x_base = data.x[:1] * 0
            data.x = x_base
            data.xg = torch.cat(super_nodes, 0).view((1, -1))

        return data

    def cloud_point_transform(self, data):
        if data.x is None:
          data.x = torch.ones([data.num_nodes, 1], dtype=torch.float)

        deg, deg_onehot = self.compute_degree(data.edge_index, data.num_nodes)
        akx = self.compute_akx(data.num_nodes, data.x, data.edge_index)
        cent = self.compute_centrality(data)
        data.x = torch.cat([data.x, data.pos, deg, deg_onehot, akx, cent], -1)

        if self.remove_edges != "none":
            if self.remove_edges == "all":
                self_edge = None
            else:  # only keep self edge
                self_edge = torch.tensor(range(data.num_nodes)).view((1, -1))
                self_edge = torch.cat([self_edge, self_edge], 0)
            data.edge_index = self_edge

        # Reduce nodes by degree-based grouping
        if self.group_degree > 0:
            assert self.remove_edges == "all", "remove all edges"
            x_base = data.x
            deg_base = deg.view(-1)
            super_nodes = []
            for k in range(1, self.group_degree + 1):
                eq_idx = deg_base == k
                gt_idx = deg_base > k
                x_to_group = x_base[eq_idx]
                x_base = x_base[gt_idx]
                deg_base = deg_base[gt_idx]
                group_size = torch.zeros([1, 1]) + x_to_group.size(0)
                if x_to_group.size(0) == 0:
                  super_nodes.append(
                      torch.cat([group_size, data.x[:1]*0], -1))
                else:
                  super_nodes.append(
                      torch.cat([group_size,
                                 x_to_group.mean(0, keepdim=True)], -1))
            if x_base.size(0) == 0:
                x_base = data.x[:1] * 0
            data.x = x_base
            data.xg = torch.cat(super_nodes, 0).view((1, -1))

        return data

    def compute_degree(self, edge_index, num_nodes):
        row, col = edge_index
        deg = degree(row, num_nodes)
        deg = deg.view((-1, 1))

        if self.onehot_maxdeg is not None and self.onehot_maxdeg > 0:
            max_deg = torch.tensor(self.onehot_maxdeg, dtype=deg.dtype)
            deg_capped = torch.min(deg, max_deg).type(torch.int64)
            deg_onehot = F.one_hot(
                deg_capped.view(-1), num_classes=self.onehot_maxdeg + 1)
            deg_onehot = deg_onehot.type(deg.dtype)
        else:
            deg_onehot = self.empty_feature(num_nodes)

        if not self.degree:
            deg = self.empty_feature(num_nodes)

        return deg, deg_onehot

    def compute_centrality(self, data):
        if not self.centrality:
          return self.empty_feature(data.num_nodes)

        G = nx.Graph(data.edge_index.numpy().T.tolist())
        G.add_nodes_from(range(data.num_nodes))  # in case missing node ids
        closeness = nx.algorithms.closeness_centrality(G)
        betweenness = nx.algorithms.betweenness_centrality(G)
        pagerank = nx.pagerank_numpy(G)
        centrality_features = torch.tensor(
            [[closeness[i], betweenness[i], pagerank[i]] for i in range(
                data.num_nodes)])
        return centrality_features

    def compute_akx(self, num_nodes, x, edge_index, edge_weight=None):
        if self.AK is None or self.AK <= 0:
            return self.empty_feature(num_nodes)

        edge_index, norm = self.norm(
            edge_index, num_nodes, edge_weight, diag_val=self.edge_norm_diag)

        xs = []
        for k in range(1, self.AK + 1):
            x = self.propagate(edge_index, x=x, norm=norm)
            xs.append(x)
        return torch.cat(xs, -1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, diag_val=1e-8, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        # Add edge_weight for loop edges.
        loop_weight = torch.full((num_nodes, ),
                                 diag_val,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def empty_feature(self, num_nodes):
        return torch.zeros([num_nodes, 0])
