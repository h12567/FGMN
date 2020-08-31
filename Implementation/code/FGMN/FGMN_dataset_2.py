import numpy as np
import os.path as osp
import pathlib
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

class FGMNDataset(InMemoryDataset):

    NUM_MSP_PEAKS = 16
    ATOM_VARIABLE = 1
    EDGE_VARIABLE = 2
    MSP_VARIABLE = 3
    EDGE_FACTOR = 4
    MSP_FACTOR = 5

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(FGMNDataset, self).__init__(root, transform,
                                          pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            "vertex_arr_sort_per.npy",
            "mol_adj_arr_sort_per.npy",
            "msp_arr_sort_per.npy",
        ]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def get_atom_nodes(self, atom_arr, node_features, node_labels):
        atom_mass_mapping = [12, 1, 16, 14]
        for atom_idx in atom_arr:
            # node_features.append([self.ATOM_VARIABLE, atom_idx, atom_mass_mapping[atom_idx]]) # + [-1] * 12)
            node_features.append([self.ATOM_VARIABLE, atom_idx, atom_idx])
            node_labels.append(-1)

    def get_edge_nodes(self, mol_adj_arr, num_atoms, node_features, node_labels, edge_idx, edge_attr,
                       factor_features, factor_labels):
        total_edges_so_far = 0
        for x in range(num_atoms):
            for y in range(x+1, num_atoms):
            # for y in range(num_atoms):
                node_features.append([self.EDGE_VARIABLE, x, y]) # + [-1] * 12)
                node_labels.append(mol_adj_arr[x][y])
                # edge_idx.append([num_atoms + x * num_atoms + y, x])
                # edge_idx.append([num_atoms + x * num_atoms + y, y])
                # edge_idx.append([num_atoms + x * num_atoms + (y - x - 1), x])
                # edge_idx.append([num_atoms + x * num_atoms + (y - x - 1), y])
                edge_idx.append([num_atoms + total_edges_so_far, x])
                edge_idx.append([num_atoms + total_edges_so_far, y])
                total_edges_so_far += 1
                # edge_idx.append([x, num_atoms + x * num_atoms + y])
                # edge_idx.append([y, num_atoms + x * num_atoms + y])
                # factor_features.append([self.EDGE_FACTOR, num_atoms+ x*num_atoms + y, x, y] + [-1] * 11)
                # factor_labels.append(-1)
                for _ in range(2):
                    edge_attr.append([1])

    def get_msp_nodes(self, msp_arr, k, num_atoms, node_features, node_labels, edge_idx, edge_attr,
                      factor_features, factor_labels):
        k_largest_idxes = np.argsort(msp_arr)[-k:]
        k_largest_peaks = msp_arr[k_largest_idxes]
        for i in range(k):
            node_features.append([self.MSP_VARIABLE, k_largest_peaks[i], k_largest_idxes[i]]) # + [-1] * 12)
            node_labels.append(-1)
            for atom_idx in range(num_atoms):
                # edge_idx.append([num_atoms + num_atoms**2 + i, atom_idx])
                edge_idx.append([num_atoms + int((num_atoms**2 - num_atoms) / 2) + i, atom_idx])
                # edge_idx.append([atom_idx, num_atoms + int((num_atoms**2 - num_atoms) / 2) + i])
                for _ in range(1):
                    edge_attr.append([2])
            # factor_features.append([self.MSP_FACTOR, num_atoms + num_atoms**2 + i] + list(range(num_atoms))
            #                        + [-1] * (13 - num_atoms))
            # factor_labels.append(-1)

    def process(self):
        # min_len: 5, max_len: 13
        atom_arr_all = np.load(pathlib.Path(self.raw_paths[0]), allow_pickle=True)

        mol_adj_arr_all = np.load(pathlib.Path(self.raw_paths[1]), allow_pickle=True)
        msp_arr_all = np.load(pathlib.Path(self.raw_paths[2]), allow_pickle=True)

        data_list = []
        for i in range(atom_arr_all.shape[0]):
            atom_arr, mol_adj_arr, msp_arr = atom_arr_all[i], mol_adj_arr_all[i], msp_arr_all[i]
            node_features, node_labels, edge_idx, edge_attr = [], [], [], []
            factor_features, factor_labels = [], []
            self.get_atom_nodes(atom_arr, node_features, node_labels)
            self.get_edge_nodes(
                mol_adj_arr, len(atom_arr), node_features, node_labels,
                edge_idx, edge_attr, factor_features, factor_labels
            )
            self.get_msp_nodes(
                msp_arr, self.NUM_MSP_PEAKS, len(atom_arr), node_features, node_labels,
                edge_idx, edge_attr, factor_features, factor_labels
            )
            # node_features = node_features + factor_features
            # node_labels = node_labels + factor_labels
            node_features = torch.FloatTensor(node_features)
            # node_labels = torch.FloatTensor(node_labels)
            node_labels = torch.LongTensor(node_labels)
            edge_idx = torch.LongTensor(edge_idx).transpose(0, 1)
            edge_attr = torch.FloatTensor(edge_attr)
            data = Data(x=node_features, edge_index=edge_idx, edge_attr=edge_attr, y=node_labels)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
