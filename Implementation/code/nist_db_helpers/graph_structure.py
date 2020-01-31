import numpy as np
import pynauty
from collections import deque
from rdkit import Chem

#------
# APSP
#-------
def floydwarshall(A):
    '''
    A is nxn with elems as edge wts.
    Returns:
      D: distances. D[i][j] is shortest distance between i and j. n x n
      p: predecessor visitation order. n x n.
         Example:
         # supose find path v id 9 to 13
         i=9
         j=14
         hops = 1
         while (p[i,j] != i):
             hops += 1
             print(i, j, p[i,j])
             j = int(p[i,j])
         print(i, j, p[i,j])
         print('hops',hops)
    '''
    V = A.shape[0]
    D = np.copy(A) #dists
    D[D<1] = np.inf
    D[list(range(V)), list(range(V))] = 0
    temp = np.repeat(np.arange(1, V+1).reshape(-1,1), repeats=V, axis=-1)
    p = np.multiply(A, temp) + np.ones_like(A)*-1 #predecssor visitation order.
    # print('A', A)
    # print('p',p)
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if D[i][j] > D[i][k] + D[k][j]: #relax
                    D[i][j] = D[i][k] + D[k][j]
                    p[i][j] = p[k][j] #update predecessor
    return D, p

#----------------
# Orbit partition
#----------------
def compute_orbits(node_list, A):
    '''
    WARN:
      node_list & A should be a connected component. Don't input multiple connected components.
    args:
      node_list e.g. ['C', 'N', 'N', 'C', 'C', 'O', 'P', 'O', 'C']
      A adj mat n x n np array. 0 edge absent. >0 edge present. Edge colors & edge weights are ignored.
    returns:
      orbit labels. 1 label per atom.
      number of orbit partitions. scalar.
    '''
    A = adj_mat_2_adj_dict(A)
    vertex_coloring = node_list_2_vertex_coloring(node_list)
    G = pynauty.Graph(number_of_vertices=len(node_list), 
                    directed=False,
                    adjacency_dict=A,
                    vertex_coloring=vertex_coloring)
    automorphism_group = pynauty.autgrp(G)  #return -> (generators, grpsize1, grpsize2, orbits, numorbits)
    # print('automorphism_group', automorphism_group)
    n_orbits = automorphism_group[-1] # e.g. 3
    orbits = automorphism_group[-2]   # e.g. [0 1 2 2 1 0]
    return orbits, n_orbits

def adj_mat_2_adj_dict(A):
    # Example:
    # A=[[0. 1. 0. 0. 0.]
    #   [1. 0. 2. 0. 0.]
    #   [0. 2. 0. 1. 0.]
    #   [0. 0. 1. 0. 1.]
    #   [0. 0. 0. 1. 0.]]
    # return:
    # {0: (1,), 1: (0, 2), 2: (1, 3), 3: (2, 4), 4: (3,)}
    adj_dict = {}
    for i in range(A.shape[0]):
        adj_dict[i] = list(np.where(A[i,:] > 0)[0])
    # print('adj_dict', adj_dict)
    return adj_dict

def node_list_2_vertex_coloring(node_list):
    atom_types = set(node_list) #{'C', 'N', 'O', 'P'}
    arr = np.array(node_list)   #['C', 'N', 'N', 'C', 'C', 'O', 'P', 'O', 'C']
    vertex_coloring = []
    for atom_type in atom_types:
        vertex_coloring.append(set(np.where(arr == atom_type)[0]))
    return vertex_coloring      # [{0, 8, 3, 4}, {5, 7}, {6}, {1, 2}]

#---------------------
# Connected Components
#---------------------
def connected_components(A, n_atoms):
    """
    Returns connected component labels.
    E.g. return [1, 2, 3, 3, 2, 4]
    means atom 0 is by itself. (labeled 1)
            atoms 1, 4 are a in a CC (labeled 2)
            atoms 2, 3 are a in a CC (labeled 3)
            atom 5 is by itself (labeled 4)
    Args:
        A: adj mat n x n. 0 edge absent. >0 edge present.
    """
    # print('A', A)
    cc_labels = [0] * n_atoms
    p = [-1] * n_atoms # parent
    #
    def DFS(v, label_give):
        stack = deque()
        stack.append(v) #start DFS at atom idx v.
        while stack: # not empty
            u = stack.pop()
            cc_labels[u] = label_give
            neighbors = (A[u,:] > 0).nonzero()
            if isinstance(neighbors,tuple):
                neighbors = neighbors[0]
            # print('neighbors of', u, ':', neighbors, neighbors.shape)
            for v in neighbors:
                v = int(v)
                if cc_labels[v] == 0:
                    p[v] = u
                    stack.append(v)
    #
    cc_idx = 0
    for v in range(n_atoms):
        if cc_labels[v] == 0:
            cc_idx += 1
            DFS(v, label_give=cc_idx)
    return cc_labels

#-------------------
# Canonical Indexing
#-------------------
def canonicalize(mol):
    # E.g.
    # returns list [2,3,1,0]
    # means atom indexed 0 in mol is mapped to canonical order 2,
    # atom indexed 1 is mapped to canonical order 3,
    # atom indexed 2 is mapped to 1,
    # atom indexed 3 is mapped to 0.
    return list(Chem.CanonicalRankAtoms(mol, breakTies=True))

def print_canonical_order(mol, f=Chem.CanonicalRankAtoms):
    order = list(f(mol, breakTies=True))
    for i, j in enumerate(order):
        print('idx', i, 'order', j, mol.GetAtomWithIdx(i).GetSymbol())
    mol = Chem.RWMol(mol)
    for i, j in enumerate(order):
        mol.GetAtomWithIdx(i).SetIntProp('molAtomMapNumber', j + 1)
    print(Chem.MolToSmiles(mol, canonical=True))

#-------------------------
# make molecule from graph
#-------------------------
def mol_from_graph(node_list, adjacency_matrix):
    # create empty editable mol object
    mol = Chem.RWMol()
    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(node_list)):
        a = Chem.Atom(node_list[i])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx
    # add bonds between adjacent atoms
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):
            # only traverse half the matrix
            if iy <= ix:
                continue
            # add relevant bond type (there are many more of these)
            if bond == 0:
                continue
            elif bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)
            elif bond == 3:
                bond_type = Chem.rdchem.BondType.TRIPLE
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    # Convert RWMol to Mol object
    # mol = mol.GetMol()            
    return mol, node_to_idx