import torch
import numpy as np
import torch_geometric
import random

from scipy import io
from torch_geometric.datasets import Planetoid
from sklearn.model_selection import train_test_split

from torch_geometric.utils import sort_edge_index, from_scipy_sparse_matrix, to_scipy_sparse_matrix, degree, contains_self_loops, remove_self_loops
import scipy.sparse as sp
import re


def load_heter_data(dataset_name):
    DATAPATH = 'data/heterophily_datasets_matlab'
    fulldata = io.loadmat(f'{DATAPATH}/{dataset_name}.mat')


    # fulldata = io.loadmat(r"https://github.com/alexfanjn/FAGCN_PyG/tree/main/data/heterophily_datasets_matlab/film.mat")

    edge_index = fulldata['edge_index']
    node_feat = fulldata['node_feat']
    label = np.array(fulldata['label'], dtype=np.int32).flatten()

    num_features = node_feat.shape[1]
    num_classes = np.max(label) + 1
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    x = torch.tensor(node_feat)
    y = torch.tensor(label, dtype=torch.long)
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    edge_index, _ = torch_geometric.utils.remove_self_loops(edge_index)
    data = torch_geometric.data.Data(x=x, edge_index=edge_index, y=y)


    return data, num_features, num_classes


def load_homo_data(dataset_name):
    dataset = Planetoid(root='/tmp/'+dataset_name, name=dataset_name)
    return dataset



def load_structure_neighborhood(embedding_file_path, num_nodes):
    new_edge_index = []
    edge_relation = []


    with open(embedding_file_path) as embedding_file:
        for line in embedding_file:
            # space : one of graph, latent_space
            # relation_type : one of 0~3
            if line.rstrip() == 'node1,node2	space	relation_type':
                continue
            line = re.split(r'[\t,]', line.rstrip())
            assert (len(line) == 4)

            new_edge_index.append([int(line[0]), int(line[1])])
            edge_relation.append(int(line[3]))

    new_edge_index = torch.tensor(np.array(new_edge_index).T)
    edge_relation = torch.tensor(np.array(edge_relation))


    new_edge_index, _ = remove_self_loops(new_edge_index)
    self_loops = torch.vstack([torch.arange(num_nodes), torch.arange(num_nodes)])
    self_loop_relation = torch.ones(num_nodes)


    new_edge_index = torch.hstack([new_edge_index, self_loops])
    edge_relation = torch.hstack([edge_relation, self_loop_relation])

    return new_edge_index, edge_relation



def set_seed(seed):

    np.random.seed(seed)
    random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
    return seed


def split_nodes(labels, train_ratio, val_ratio, test_ratio, random_state, split_by_label_flag):
    idx = torch.arange(labels.shape[0])
    if split_by_label_flag:
        idx_train, idx_test = train_test_split(idx, random_state=random_state, train_size=train_ratio+val_ratio, test_size=test_ratio, stratify=labels)
    else:
        idx_train, idx_test = train_test_split(idx, random_state=random_state, train_size=train_ratio+val_ratio, test_size=test_ratio)

    if val_ratio:
        labels_train_val = labels[idx_train]
        if split_by_label_flag:
            idx_train, idx_val = train_test_split(idx_train, random_state=random_state, train_size=train_ratio/(train_ratio+val_ratio), stratify=labels_train_val)
        else:
            idx_train, idx_val = train_test_split(idx_train, random_state=random_state, train_size=train_ratio/(train_ratio+val_ratio))
    else:
        idx_val = None

    return idx_train, idx_val, idx_test


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item()*1.0/len(labels)




# This function is generated from the following link: https://github.com/EdisonLeeeee/GreatX/blob/master/greatx/utils/modification.py
def remove_edges(edge_index, edges_to_remove):
    edges_to_remove = torch.cat(
            [edges_to_remove, edges_to_remove.flip(0)], dim=1)
    edges_to_remove = edges_to_remove.to(edge_index)

    # it's not intuitive to remove edges from a graph represented as `edge_index`
    edge_weight_remove = torch.zeros(edges_to_remove.size(1)) - 1e5
    edge_weight = torch.cat(
        [torch.ones(edge_index.size(1)), edge_weight_remove], dim=0)
    edge_index = torch.cat([edge_index, edges_to_remove], dim=1).cpu().numpy()
    adj_matrix = sp.csr_matrix(
        (edge_weight.cpu().numpy(), (edge_index[0], edge_index[1])))
    adj_matrix.data[adj_matrix.data < 0] = 0.
    adj_matrix.eliminate_zeros()
    edge_index, _ = from_scipy_sparse_matrix(adj_matrix)
    return edge_index


def edge_index_to_sparse_tensor_adj(edge_index, num_nodes):
    sparse_adj_adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
    values = sparse_adj_adj.data
    indices = np.vstack((sparse_adj_adj.row, sparse_adj_adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = sparse_adj_adj.shape
    sparse_adj_adj_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    return sparse_adj_adj_tensor



def gcn_norm(edge_index, num_nodes, device):
    a1 = edge_index_to_sparse_tensor_adj(edge_index, num_nodes).to(device)
    d1_adj = torch.diag(degree(edge_index[0], num_nodes=num_nodes)).to_sparse()
    d1_adj = torch.pow(d1_adj, -0.5)

    return torch.sparse.mm(torch.sparse.mm(d1_adj, a1), d1_adj)


