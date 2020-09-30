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
from fgmn_layer import FGNet, ValenceNet, MULTIPLY_MODE, ADDITION_MODE
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

b = 64

train_loader = DataLoader(dataset, batch_size=b, shuffle=False)

def accuracy(pred, target, hydro_atom_idxes, x, edge_indicator_idx):
    total_match = 0
    total_zeros = 0
    for i in range(len(pred)):
        if pred[i][0].item() == target[i][0].item():
            total_match += 1
        if target[i][0].item() == 0:
            total_zeros += 1
    return total_match / len(pred), total_zeros / len(pred)

def get_acc_helper(edges_nodes, edge_indicator_idx, valid_labels, num_real_atoms, x, data, adj):
    edges_nodes_out = F.log_softmax(edges_nodes, dim=-1)

    atom_idxes = torch.flatten((x[:, 0] == ATOM_VARIABLE).nonzero())
    atoms = x[atom_idxes]
    hydro_atom_idxes = torch.flatten((atoms[:, 1] == 1).nonzero())

    valid_out_init = edges_nodes_out[edge_indicator_idx]
    valid_pred_init = torch.argmax(valid_out_init, dim=-1)
    # num_helper = int((num_real_atoms ** 2 - num_real_atoms) / 2)

    acc = accuracy(
        valid_pred_init, valid_labels, hydro_atom_idxes, x, edge_indicator_idx
    )
    return acc

def get_count_real_atoms(x):
    atom_idxes = torch.flatten((x[:, 0] == ATOM_VARIABLE).nonzero())
    atoms = x[atom_idxes]
    valid_atom_idxes = torch.flatten((atoms[:, 1] != 1).nonzero())
    return len(valid_atom_idxes)

def noise_and_decode(
        x, out_edges, observations, fact_l_A, edge_indicator_idx,
        valid_labels, result_dict, data, adj, noise_level=0
):
    print("NOISE LEVEL: %s" % str(noise_level))
    result_dict[noise_level] = dict()

    num_real_atoms = get_count_real_atoms(x)

    observations = observations.clone()
    observations += torch.tensor(np.random.exponential(noise_level, observations.shape)).cuda()

    edges_nodes_init = observations.clone()
    # edges_nodes_init = observations.clone()
    edges_nodes_observed = observations.clone()
    init_acc = get_acc_helper(edges_nodes_observed, edge_indicator_idx,
                              valid_labels, num_real_atoms, x, data, adj)
    result_dict[noise_level][0] = init_acc
    print("ACCURACY BEFORE: %s" %str(init_acc[0]))

    fA_valence = ValenceNet()

    ###### EXP 1: Normal Sum Messages, One Iteration ###############################
    all_msgs_to_edges_nodes = None
    for i in range(len(fact_l_A)):
        if fact_l_A[i] is not None:
            msg_to_edge = fA_valence.compute(
                x, edges_nodes_init, fact_l_A[i], combine_mode=ADDITION_MODE,
            )
            if all_msgs_to_edges_nodes is not None:
                # raise Exception("Cannot have more than one factor length for single molecule")
                all_msgs_to_edges_nodes = torch.cat([all_msgs_to_edges_nodes, msg_to_edge])
            else:
                all_msgs_to_edges_nodes = msg_to_edge
    edges_nodes_after = edges_nodes_init + observations * all_msgs_to_edges_nodes.sum(dim=0)
    after_acc = get_acc_helper(edges_nodes_after, edge_indicator_idx, valid_labels,
                               num_real_atoms, x, data, adj)
    result_dict[noise_level][1] = after_acc
    print("ACCURACY AFTER 1: %s" %str(after_acc[0]))
    #################################################################################

    ###### EXP 2: Normal Multiply Messages, One Iteration ###############################
    all_msgs_to_edges_nodes = None
    for i in range(len(fact_l_A)):
        if fact_l_A[i] is not None:
            msg_to_edge = fA_valence.compute(
                x, edges_nodes_init, fact_l_A[i], combine_mode=MULTIPLY_MODE,
            )
            if all_msgs_to_edges_nodes is not None:
                # raise Exception("Cannot have more than one factor length for single molecule")
                all_msgs_to_edges_nodes = torch.cat([all_msgs_to_edges_nodes, msg_to_edge])
            else:
                all_msgs_to_edges_nodes = msg_to_edge
    edges_nodes_after = edges_nodes_init + observations * all_msgs_to_edges_nodes.sum(dim=0)
    after_acc = get_acc_helper(edges_nodes_after, edge_indicator_idx,
                               valid_labels, num_real_atoms, x, data, adj)
    result_dict[noise_level][2] = after_acc
    print("ACCURACY AFTER 2: %s" %str(after_acc[0]))
    #################################################################################

    # ###### EXP 3: Normal Sum Messages, 3 Iterations ###############################
    # all_msgs_to_edges_nodes = None
    # edges_nodes_after = edges_nodes_init.clone()
    # for _ in range(3):
    #     for i in range(len(fact_l_A)):
    #         if fact_l_A[i] is not None:
    #             msg_to_edge = fA_valence.compute(
    #                 x, edges_nodes_init, fact_l_A[i], combine_mode=ADDITION_MODE,
    #             )
    #             if all_msgs_to_edges_nodes is not None:
    #                 # raise Exception("Cannot have more than one factor length for single molecule")
    #                 all_msgs_to_edges_nodes = torch.cat([all_msgs_to_edges_nodes, msg_to_edge])
    #             else:
    #                 all_msgs_to_edges_nodes = msg_to_edge
    #     edges_nodes_after += observations * all_msgs_to_edges_nodes.sum(dim=0)
    # after_acc = get_acc_helper(edges_nodes_after, edge_indicator_idx, valid_labels, num_real_atoms)
    # result_dict[noise_level][3] = after_acc
    # print("ACCURACY AFTER 3: %s" % str(after_acc[0]))
    # #################################################################################
    #
    # ###### EXP 3: Normal Multiply Messages, 3 Iterations ###############################
    # all_msgs_to_edges_nodes = None
    # edges_nodes_after = edges_nodes_init.clone()
    # for _ in range(3):
    #     for i in range(len(fact_l_A)):
    #         if fact_l_A[i] is not None:
    #             msg_to_edge = fA_valence.compute(
    #                 x, edges_nodes_init, fact_l_A[i], combine_mode=MULTIPLY_MODE,
    #             )
    #             if all_msgs_to_edges_nodes is not None:
    #                 # raise Exception("Cannot have more than one factor length for single molecule")
    #                 all_msgs_to_edges_nodes = torch.cat([all_msgs_to_edges_nodes, msg_to_edge])
    #             else:
    #                 all_msgs_to_edges_nodes = msg_to_edge
    #     edges_nodes_after += observations * all_msgs_to_edges_nodes.sum(dim=0)
    # after_acc = get_acc_helper(edges_nodes_after, edge_indicator_idx, valid_labels, num_real_atoms)
    # result_dict[noise_level][4] = after_acc
    # print("ACCURACY AFTER 4: %s" % str(after_acc[0]))
    # #################################################################################

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
                data.x, None, None, None, None, data.edge_index_2, data.edge_attr_2,
            )
            bond_type = 4
            out_edges = torch.zeros(data.x.size(0), bond_type).cuda()
            edge_indicator_idx = (torch.Tensor([x[0] for x in data.x]) == EDGE_VARIABLE).nonzero()

            atom_idxes = torch.flatten((data.x[:, 0] == ATOM_VARIABLE).nonzero())
            atoms = data.x[atom_idxes]
            hydro_atom_idxes = torch.flatten((atoms[:, 1] == 1).nonzero())

            a = []
            for i, idx in enumerate(edge_indicator_idx[:, 0]):
                val = adj[idx][hydro_atom_idxes].sum()
                if val == 0:
                    a.append(i)

            edge_indicator_idx = edge_indicator_idx[a]
            valid_labels = data.y[edge_indicator_idx]
            observations = out_edges.clone()
            observations[edge_indicator_idx[:, 0]] = torch.eye(bond_type)[valid_labels[:, 0]].cuda()

            result_dict = dict()
            noise_level_l = [0.1, 0.5, 1, 2, 5, 10]
            for noise_level in noise_level_l:
                noise_and_decode(
                    data.x, out_edges, observations, fact_l_A,
                    edge_indicator_idx, valid_labels, result_dict, data, adj, noise_level=noise_level
                )
            i += 1
        except Exception as e:
            print(e)

experiment()
