import torch
import numpy as np

def valence_compute():
    pass

ATOM_VARIABLE = 1
EDGE_VARIABLE = 2
MSP_VARIABLE = 3
EDGE_FACTOR = 4
MSP_FACTOR = 5
EDGEATOM_EDGE_INDEX = 1
MSPATOM_EDGE_INDEX = 2
EDGEEDGE_EDGE_INDEX = 3

def get_edgeatomfactorsntypes(adj, dim, bond_type, nodes, edge_index_2, edge_attr_2):

    fact = [] # (number of factors , 3) 3 here are [edge_index, atom_1_index, atom_2_index]
    fact_dim = []

    # edge_variable_idxes = (nodes[:, 0] == EDGE_VARIABLE).nonzero()
    # atom_variable_idxes = (nodes[:, 0] == ATOM_VARIABLE).nonzero()
    # a = edge_index_2[0, torch.flatten(edge_variable_idxes)].cpu()
    # b = edge_index_2[1, torch.flatten(atom_variable_idxes)].cpu()
    # res_np = np.intersect1d(a, b)
    # res = torch.from_numpy(res_np)

    edgeatom_edge_idxes = torch.flatten((edge_attr_2[:, 0] == EDGEATOM_EDGE_INDEX).nonzero())

    edge_index_short = edge_index_2[:, edgeatom_edge_idxes]

    fact = None

    i = 0

    # Todo: Factoring this
    # for i in range(edge_index_short.shape[1] - 1):
    while i < (edge_index_short.shape[1] - 1):

        tmp = torch.stack(
            [
                edge_index_short[0][i],
                edge_index_short[1][i],
                edge_index_short[1][i + 1]
            ]
        )

        if fact is not None:
            fact = torch.cat([fact, tmp.view(1, -1)])
        else:
            fact = tmp.view(1, -1)

        fact_dim.append([
            bond_type,
            dim,
            dim
        ])

        i += 2

    # edge_factor_index = torch.flatten((nodes[:, 0] == EDGE_FACTOR).nonzero())
    #
    # fact = nodes[edge_factor_index][:, 1:4]
    # fact = nodes[edge_factor_index][:, 1:4]

    fact_l = [fact]
    fact_dim_l = [fact_dim]
    return fact_l

def get_mspatomfactorsntypes(adj, dim, bond_type, nodes, edge_index_2, edge_attr_2):
    mspatom_edge_idxes = torch.flatten((edge_attr_2[:, 0] == MSPATOM_EDGE_INDEX).nonzero())

    edge_index_short = edge_index_2[:, mspatom_edge_idxes]

    fact_l = [None] * (14+3-6+1)
    fact_dim_l = []

    i = 0
    cur_msp_node_idx = edge_index_short[0][i]

    while i < (edge_index_short.shape[1] - 1):

        atom_node_idx_arr = []
        while i < (edge_index_short.shape[1] - 1) and edge_index_short[0][i] == cur_msp_node_idx:
            atom_node_idx_arr.append(edge_index_short[1][i])
            i += 1

        tmp = torch.stack(
            [cur_msp_node_idx] + atom_node_idx_arr
        )

        fact_len_idx = len(atom_node_idx_arr) - 5

        if fact_l[fact_len_idx] is not None:
            fact_l[fact_len_idx] = torch.cat([fact_l[fact_len_idx], tmp.view(1, -1)])
        else:
            fact_l[fact_len_idx] = tmp.view(1, -1)

        cur_msp_node_idx = edge_index_short[0][i]

        # i += 1

    # output: list of fact, where each fact is (num_factors, 6 -> 14)
    return fact_l

def get_edgesedgesfactorsnttypes(x, adj, dim, bond_type, nodes, edge_index_2, edge_attr_2):
    atom_valence_mapping = {
        0: 4,  # Carbon
        1: 1,  # Hydrogen
        2: 2,  # Oxygen
        3: 3,  # Nitrogen
    }
    edgeedge_edge_idxes = torch.flatten((edge_attr_2[:, 0] == EDGEEDGE_EDGE_INDEX).nonzero())

    edge_index_short = edge_index_2[:, edgeedge_edge_idxes]

    sort_idx = torch.sort(edge_index_short[0, :])

    edge_index_short_2 = edge_index_short[:, sort_idx.indices]

    fact_l = [None] * (12 + 3 - 4 + 1)

    i = 0
    cur_atom_node_idx = edge_index_short_2[0][i]

    while i < (edge_index_short_2.shape[1] - 1):

        atom_node_idx_arr = []
        while i < (edge_index_short_2.shape[1]) and edge_index_short_2[0][i] == cur_atom_node_idx:
            atom_node_idx_arr.append(edge_index_short_2[1][i])
            i += 1

        tmp = torch.stack(
            [cur_atom_node_idx] + atom_node_idx_arr
        )

        fact_len_idx = len(atom_node_idx_arr) - 4

        # check if not hydrogen
        if x[cur_atom_node_idx][1] != 1:
            if fact_l[fact_len_idx] is not None:
                fact_l[fact_len_idx] = torch.cat([fact_l[fact_len_idx], tmp.view(1, -1)])
            else:
                fact_l[fact_len_idx] = tmp.view(1, -1)

        if i < (edge_index_short_2.shape[1]):
            cur_atom_node_idx = edge_index_short_2[0][i]

    return fact_l
