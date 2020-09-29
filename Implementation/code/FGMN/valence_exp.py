import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Sequential, Linear, ReLU, GRU

from torch_geometric.data import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops, to_dense_adj
import torch_geometric.transforms as T
import matplotlib.pyplot as plt

from FGMN_dataset_2 import FGMNDataset
from fgmn_layer import FGNet, ValenceNet
import utils

NUM_MSP_PEAKS = 16
ATOM_VARIABLE = 1
EDGE_VARIABLE = 2
MSP_VARIABLE = 3
EDGE_FACTOR = 4
MSP_FACTOR = 5

class Complete(object):

    def __call__(self, data):
        device = data.edge_index.device
        data.edge_attr_2 = data.edge_attr.clone()
        data.edge_index_2 = data.edge_index.clone()

        # row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        # col = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        #
        # row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        # col = col.repeat(data.num_nodes)
        # edge_index = torch.stack([row, col], dim=0)
        #
        # edge_attr = None
        # if data.edge_attr is not None:
        #     idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
        #     size = list(data.edge_attr.size())
        #     size[0] = data.num_nodes * data.num_nodes
        #     edge_attr = data.edge_attr.new_zeros(size)
        #     edge_attr[idx] = data.edge_attr
        #
        # edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        # data.edge_attr = edge_attr
        # data.edge_index = edge_index

        return data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(3)

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'FGMN')
transform = T.Compose([Complete()])
dataset = FGMNDataset(path, transform=transform).shuffle()

b = 1

train_loader = DataLoader(dataset, batch_size=b, shuffle=False)

def accuracy(pred, target):
    total_match = 0
    total_zeros = 0
    for i in range(len(pred)):
        if pred[i][0].item() == target[i][0].item():
            total_match += 1
        if target[i][0].item() == 0:
            total_zeros += 1
    return total_match / len(pred), total_zeros / len(pred)

def get_acc_helper(edges_nodes, edge_indicator_idx, valid_labels):
    edges_nodes_out = F.log_softmax(edges_nodes, dim=-1)
    valid_out_init = edges_nodes_out[edge_indicator_idx]
    valid_pred_init = torch.argmax(valid_out_init, dim=-1)
    acc = accuracy(valid_pred_init, valid_labels)
    return acc

def noise_and_decode(x, out_edges, observations, fact_l_A, edge_indicator_idx, valid_labels, noise_level=0):
    print("NOISE LEVEL: %s" % str(noise_level))

    edges_nodes_init = observations.clone()
    # edges_nodes_init = observations.clone()
    edges_nodes_observed = observations.clone()
    init_acc = get_acc_helper(edges_nodes_observed, edge_indicator_idx, valid_labels)
    print("ACCURACY BEFORE: %s" %str(init_acc[0]))

    fA_valence = ValenceNet()

    ###### EXP 1: Normal Sum Messages, One Iteration ###############################
    all_msgs_to_edges_nodes = None
    for i in range(len(fact_l_A)):
        if fact_l_A[i] is not None:
            msg_to_edge = fA_valence.compute(
                x, edges_nodes_init, fact_l_A[i]
            )
            if all_msgs_to_edges_nodes is not None:
                raise Exception("Cannot have more than one factor length for single molecule")
            else:
                all_msgs_to_edges_nodes = msg_to_edge
    edges_nodes_after = edges_nodes_init + observations * all_msgs_to_edges_nodes[0]
    after_acc = get_acc_helper(edges_nodes_after, edge_indicator_idx, valid_labels)
    print("ACCURACY AFER: %s" %str(after_acc[0]))
    #################################################################################

def experiment():
    i = 0
    for data in train_loader:
        try:
            if i >= 1:
                break
            data = data.to(device)
            edge_attr = data.edge_attr_2[:, :4].contiguous()
            adj = to_dense_adj(data.edge_index_2, batch=None,
                               edge_attr=edge_attr.argmax(-1) + 1).squeeze(0)
            fact_l_A = utils.get_edgesedgesfactorsnttypes(
                None, None, None, None, data.edge_index_2, data.edge_attr_2,
            )
            bond_type = 4
            out_edges = torch.zeros(data.x.size(0), bond_type).cuda()
            edge_indicator_idx = (torch.Tensor([x[0] for x in data.x]) == EDGE_VARIABLE).nonzero()
            valid_labels = data.y[edge_indicator_idx]
            observations = out_edges.clone()
            observations[edge_indicator_idx[:, 0]] = torch.eye(bond_type)[valid_labels[:, 0]].cuda()
            noise_and_decode(data.x, out_edges, observations, fact_l_A, edge_indicator_idx, valid_labels)
            i += 1
        except Exception as e:
            print(e)

experiment()
