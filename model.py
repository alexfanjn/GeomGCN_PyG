import torch
import torch.nn.functional as F
from utils import gcn_norm, edge_index_to_sparse_tensor_adj


class GeomGCN_layer(torch.nn.Module):
    def __init__(self, data, edge_relation, norm_adjs, num_in, num_out, device):
        super(GeomGCN_layer, self).__init__()
        self.data = data
        self.edge_relation = edge_relation
        self.norm_adjs = norm_adjs
        self.device = device

        self.linear1 = torch.nn.Linear(num_in*4, num_out)

        relation_adjs = []
        for i in range(4):
            current_relation_edge_ids = torch.where(self.edge_relation == i)[0]

            current_relation_adj_tensor = edge_index_to_sparse_tensor_adj(self.data.edge_index[:, current_relation_edge_ids], data.x.shape[0]).to(self.device)
            relation_adjs.append(current_relation_adj_tensor)

        self.relation_adjs = relation_adjs


    def forward(self, h):

        h0 = torch.sparse.mm(torch.mul(self.relation_adjs[0], self.norm_adjs), h)
        h1 = torch.sparse.mm(torch.mul(self.relation_adjs[1], self.norm_adjs), h)
        h2 = torch.sparse.mm(torch.mul(self.relation_adjs[2], self.norm_adjs), h)
        h3 = torch.sparse.mm(torch.mul(self.relation_adjs[3], self.norm_adjs), h)

        h = torch.hstack([h0, h1, h2, h3])
        h = self.linear1(h)
        return h



class GeomGCN(torch.nn.Module):
    def __init__(self, data, edge_relation, num_features, num_hidden, num_classes, dropout, layer_num=2, device='cpu'):
        super(GeomGCN, self).__init__()

        self.linear1 = torch.nn.Linear(num_features, num_hidden)

        self.edge_relation = edge_relation
        self.dropout = dropout
        self.layer_num = layer_num
        self.data = data
        self.device = device


        self.norm_adjs = gcn_norm(self.data.edge_index, self.data.y.shape[0], self.device)
        self.geomgcn_layer_1 = GeomGCN_layer(self.data, self.edge_relation, self.norm_adjs, num_features, num_hidden, self.device)
        self.geomgcn_layer_2 = GeomGCN_layer(self.data, self.edge_relation, self.norm_adjs, num_hidden, num_classes, self.device)



    def forward(self):
        h = self.geomgcn_layer_1(self.data.x)
        h = self.geomgcn_layer_2(h)
        return F.log_softmax(h, 1)

