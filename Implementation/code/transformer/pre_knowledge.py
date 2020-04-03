import numpy as np
import copy

def count_edges_dict(edge_row, vertex_arr, outgoing_edges):
    '''
    :param outgoing_edges: [(atom_type, bond_type)]
    return edges_cnt_dict: [(index_in_vertex_arr, bond_type)]
    '''
    edges_cnt_dict = []
    outgoing_edges = copy.deepcopy(outgoing_edges)
    for i, bond_type in enumerate(edge_row):
        if i < len(vertex_arr):
            data_pair = (vertex_arr[i], bond_type)
            if data_pair in outgoing_edges:
                outgoing_edges.remove(data_pair)
                edges_cnt_dict.append((i, bond_type))
    return edges_cnt_dict

def update_new_E(new_E, current_idx, edges_cnt_dict):
    for idx_in_vertex_arr, bond_type in edges_cnt_dict:
        new_E[idx_in_vertex_arr, current_idx] = bond_type
        new_E[current_idx, idx_in_vertex_arr] = bond_type
    return new_E

def generate_pre_knowledge_adj_mat(vertex_arr, num_extra_nodes, E, pre_knowledge):
    new_E = np.zeros((np.array(E).shape[0] + num_extra_nodes, np.array(E).shape[1] + num_extra_nodes)) # 3 is let's say 3 msp nodes
    current_idx = 0
    for pre_atom_type, outgoing_edges in pre_knowledge:
        satisfied = False
        while current_idx < len(vertex_arr):
            atom_type = vertex_arr[current_idx]
            if atom_type == pre_atom_type:
                edges_cnt_dict = count_edges_dict(E[current_idx, :], vertex_arr, outgoing_edges)
                if len(edges_cnt_dict) == len(outgoing_edges):
                    new_E = update_new_E(new_E, current_idx, edges_cnt_dict)
                    satisfied = True
                    break
            current_idx += 1
        assert satisfied == True

    # set entries between atom nodes and msp nodes all to 5
    new_E[-num_extra_nodes:, :] = np.full((num_extra_nodes, new_E.shape[1]), 5)
    new_E[:, -num_extra_nodes:] = np.full((new_E.shape[0], num_extra_nodes), 5)
    return new_E

if __name__ == "__main__":
    vertex_arr = [0, 0, 2, 2]
    pre_knowledge = [(0, [(2, 1), (2, 2)])]
    E = [
        [0, 0, 2, 1],
        [0, 0, 1, 0],
        [2, 1, 0, 0],
        [1, 0, 0, 0],
    ]
    num_extra_nodes = 3
    new_E = generate_pre_knowledge_adj_mat(np.array(vertex_arr), num_extra_nodes, np.array(E), np.array(pre_knowledge))
    expected_new_E = [
       [ 0.,  0.,  2.,  1., -1., -1., -1.],
       [ 0.,  0.,  0.,  0., -1., -1., -1.],
       [ 2.,  0.,  0.,  0., -1., -1., -1.],
       [ 1.,  0.,  0.,  0., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1.],
       [-1., -1., -1., -1., -1., -1., -1.]
    ]
    assert np.array_equal(new_E, expected_new_E)
