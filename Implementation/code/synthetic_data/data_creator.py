import numpy as np

vertex_arr = np.load("base_data/vertex_arr_sort_per.npy", allow_pickle=True) #1843
mol_adj_arr = np.load("base_data/mol_adj_arr_sort_per.npy", allow_pickle=True)
msp_arr = np.load("base_data/msp_arr_sort_per.npy", allow_pickle=True)

atom_mass = [12, 1, 16, 14]

a = 1