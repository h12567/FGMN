import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
d_n=800
atom_mass = [12,1,16,14,31,32]
'''
mass_spectrum relative y
type=1: (13,k+13+atoms_type) 800-->(1,k)
type=2: (13,k+13+atoms_type) 800-->(13,k)
type=3: (13+k,13+atoms_type) 800-->(k,8)
type=4: (13+k,13+atoms_type) topk
type=5: (13+k,13+atoms_type) topk x value
type=6: 1d cnn input atom_type, atom_position and atom_mass, x and y values of mass_spectrum
(5,1,d_model)
type=7: (13+k, atom_mass(1)+atom_pos) atom_mass embedding+type one hot embedding, peak_mass embedding+all 0
'''
def GetInput(vertex_arr, msp_arr, max_atoms, atoms_type, d_model=32, k=30, type=1):
    atom_num = torch.zeros((len(vertex_arr)),dtype=torch.int64)
    if type!=7:
        new_vertex = torch.zeros((len(vertex_arr), max_atoms, max_atoms+atoms_type),dtype=torch.int64)  # N,13,6+13=19
        for i in range(len(vertex_arr)):
            atom_num[i] = len(vertex_arr[i])
            for j in range(len(vertex_arr[i])):
                new_vertex[i, j, 0:atoms_type] = torch.tensor([1 if ii==vertex_arr[i][j] else 0 for ii in range(atoms_type)])
                new_vertex[i,j,atoms_type:] = torch.tensor([1 if ii==j else 0 for ii in range(max_atoms)])
    else:
        new_vertex = torch.zeros((len(vertex_arr), max_atoms, max_atoms +1),
                                 dtype=torch.int64)  # N,13,1+13=14
        for i in range(len(vertex_arr)):
            atom_num[i] = len(vertex_arr[i])
            for j in range(len(vertex_arr[i])):
                new_vertex[i, j, 0:1] = torch.tensor([atom_mass[int(vertex_arr[i][j])]])
                new_vertex[i, j, 1:] = torch.tensor([1 if ii == j else 0 for ii in range(max_atoms)])

    if type==1:
        new_inputs = torch.zeros((len(vertex_arr), max_atoms, atoms_type + max_atoms + k+1),dtype=torch.float32)  # [N,13,13+4+k=47]
        new_inputs[:,:,0:max_atoms+atoms_type]=new_vertex
        embedding = nn.Embedding(d_n,k,padding_idx=0)
        new_msp = embedding(msp_arr) #(N,1,800,k)
        new_msp = torch.sum(new_msp,dim=1) #(N,k)
        new_msp = F.normalize(new_msp,p=2,dim=1) #normalized msp
        for i in range(len(vertex_arr)):
            new_inputs[i,:len(vertex_arr[i]),atoms_type+max_atoms:-1] = new_msp[i].repeat(len(vertex_arr[i]),1) #(length, 17+k=47)
    elif type==2:
        new_inputs = torch.zeros((len(vertex_arr), max_atoms, atoms_type + max_atoms + k+1),
                                 dtype=torch.float32)  # [N,13,17+k=47]
        new_inputs[:, :, 0:max_atoms + atoms_type] = new_vertex
        for i in range(max_atoms):
            embedding = nn.Embedding(d_n, k, padding_idx=0)
            new_msp = embedding(msp_arr)  # (N,1,800,k)
            new_msp = torch.sum(new_msp, dim=1)  # (N,k)
            new_msp = F.normalize(new_msp, p=2, dim=1)  # normalized msp
            for j in range(len(vertex_arr)):
                if len(vertex_arr[j])>=i+1:
                    new_inputs[j, 0:len(vertex_arr[j]), atoms_type + max_atoms:-1] = new_msp[i].repeat(len(vertex_arr[j]),
                                                                                1)  # (length, 17+k=47)
    elif type==3:
        new_inputs = torch.zeros((len(vertex_arr), max_atoms+k, atoms_type + max_atoms),
                                 dtype=torch.float32)  # [N,13+k,17]
        new_inputs[:, 0:13,:] = new_vertex
        for i in range(k):
            embedding = nn.Embedding(d_n, max_atoms+atoms_type, padding_idx=0)
            new_msp = embedding(msp_arr)  # (N,1,800,17)
            new_msp = torch.sum(new_msp, dim=1)  # (N,17)
            new_msp = F.normalize(new_msp, p=2, dim=1) # normalized msp (N,1,17)
            new_inputs[:,max_atoms+i, :] = new_msp
    elif type==4:
        new_inputs = torch.zeros((len(vertex_arr), max_atoms + k, d_model*(max_atoms+atoms_type)),
                                 dtype=torch.float32)  # [N,13+k,304]
        embedding1 = nn.Embedding(2, d_model)
        embed_vertex = embedding1(new_vertex).view(len(vertex_arr),max_atoms,-1) #[225, 13, 304]
        new_inputs[:, 0:13, :] = F.normalize(embed_vertex,dim=2)
        new_msp = torch.zeros((len(msp_arr),k),dtype=torch.int64) #[N,k]
        for i in range(len(msp_arr)):
            peaks = []
            for j in range(1, 800 - 1):
                if msp_arr[i, j - 1] < msp_arr[i, j] and msp_arr[i, j + 1] < msp_arr[i, j]: peaks.append(msp_arr[i,j])
            peaks.sort(reverse=True)  # descending
            while (len(peaks) < k):
                peaks.append(0)
            new_msp[i] = torch.tensor(peaks[:k], dtype=torch.int64)
        embedding2 = nn.Embedding(1000, d_model*(max_atoms+atoms_type))
        new_msp = embedding2(new_msp)  # (N,1,k,17)
        new_msp = F.normalize(new_msp, p=2, dim=2)  # normalized msp
        new_inputs[:, max_atoms:, :] = new_msp
    elif type == 5:
        new_inputs = torch.zeros((len(vertex_arr), max_atoms + k, d_model * (max_atoms + atoms_type)),
                                 dtype=torch.float32)  # [N,13+k,304]
        new_inputs[:, 0:13, :max_atoms + atoms_type] = new_vertex
        embedding1 = nn.Embedding(2, d_model-1)
        embed_vertex = embedding1(new_vertex).view(len(vertex_arr), max_atoms, -1)  # [225, 13, 304]
        new_inputs[:, 0:13, max_atoms + atoms_type:] = F.normalize(embed_vertex, dim=2)
        new_msp = torch.zeros((len(msp_arr), k), dtype=torch.int64)  # [N,k]
        for i in range(len(msp_arr)):
            peaks = []
            for j in range(1, 800 - 1):
                if msp_arr[i, j - 1] < msp_arr[i, j] and msp_arr[i, j + 1] < msp_arr[i, j]: peaks.append(j)
            peaks.sort(reverse=True)  # descending
            while (len(peaks) < k):
                peaks.append(0)
            new_msp[i] = torch.tensor(peaks[:k], dtype=torch.int64)
        embedding2 = nn.Embedding(800, d_model * (max_atoms + atoms_type))
        new_msp = embedding2(new_msp)  # (N,1,k,17)
        new_msp = F.normalize(new_msp, p=2, dim=2)  # normalized msp
    elif type == 6:
        new_inputs = torch.zeros((len(vertex_arr), max_atoms + k, d_model * (max_atoms + atoms_type)),
                                 dtype=torch.float32)  # [N,13+k,304]
        new_inputs[:, 0:13, :max_atoms + atoms_type] = new_vertex[:-1]
        embedding1 = nn.Embedding(2, d_model - 1)
        embed_vertex = embedding1(new_vertex).view(len(vertex_arr), max_atoms, -1)  # [225, 13, 304]
        new_inputs[:, 0:13, max_atoms + atoms_type:] = F.normalize(embed_vertex, dim=2)
        new_msp = torch.zeros((len(msp_arr), k), dtype=torch.int64)  # [N,k]
        for i in range(len(msp_arr)):
            peaks = []
            for j in range(1, 800 - 1):
                if msp_arr[i, j - 1] < msp_arr[i, j] and msp_arr[i, j + 1] < msp_arr[i, j]: peaks.append(j)
            peaks.sort(reverse=True)  # descending
            while (len(peaks) < k):
                peaks.append(0)
            new_msp[i] = torch.tensor(peaks[:k], dtype=torch.int64)
        embedding2 = nn.Embedding(800, d_model * (max_atoms + atoms_type))
        new_msp = embedding2(new_msp)  # (N,1,k,17)
        new_msp = F.normalize(new_msp, p=2, dim=2)  # normalized msp
        new_inputs[:, max_atoms:, :] = new_msp
        #[225, 43, 304]
    elif type==7:
        new_inputs = torch.zeros((len(vertex_arr), max_atoms + k, d_model * (max_atoms + atoms_type)),
                                 dtype=torch.float32)  # [N,13+k,304]
        new_inputs[:, 0:13, d_model * (max_atoms + atoms_type)-max_atoms:] = new_vertex[:,:,1:]
        embedding1 = nn.Embedding(atoms_type, d_model * (max_atoms + atoms_type)-max_atoms)
        embed_vertex = embedding1(new_vertex[:,:,1]).view(len(vertex_arr), max_atoms, -1)  # [225, 13, d_model*10-max_atoms]
        new_inputs[:, 0:13, :d_model * (max_atoms + atoms_type)-max_atoms] = F.normalize(embed_vertex, dim=2)
        new_msp = torch.zeros((len(msp_arr), k), dtype=torch.int64)  # [N,k]
        for i in range(len(msp_arr)):
            peaks = []
            for j in range(1, 800 - 1):
                if msp_arr[i, j - 1] < msp_arr[i, j] and msp_arr[i, j + 1] < msp_arr[i, j]: peaks.append(j)
            peaks.sort(reverse=True)  # descending
            while (len(peaks) < k):
                peaks.append(0)
            new_msp[i] = torch.tensor(peaks[:k], dtype=torch.int64)
        embedding2 = nn.Embedding(800, d_model * (max_atoms + atoms_type)-max_atoms)
        new_msp = embedding2(new_msp)  # (N,1,k,17)
        new_msp = F.normalize(new_msp, p=2, dim=2)  # normalized msp
        new_inputs[:, max_atoms:, :d_model * (max_atoms + atoms_type)-max_atoms] = new_msp
        new_inputs[:, max_atoms:, d_model * (max_atoms + atoms_type) - max_atoms:] = torch.zeros((max_atoms),dtype=torch.float32)
    return new_inputs, atom_num

def getBondNum(mol_adj_arr,max_atoms):
    mol_adj_arr= torch.tensor(mol_adj_arr, dtype=torch.int64)
    bond_num = [0, 0, 0, 0]
    for i in range(len(mol_adj_arr)):
        for m in range(max_atoms):
            for n in range(max_atoms):
                if mol_adj_arr[i][m][n] == 0:
                    bond_num[0] += 1
                elif mol_adj_arr[i, m, n] == 1:
                    bond_num[1] += 1
                elif mol_adj_arr[i, m, n] == 2:
                    bond_num[2] += 1
                elif mol_adj_arr[i, m, n] == 3:
                    bond_num[3] += 1
    for i in range(len(bond_num)):
        bond_num[i] = 1/bond_num[i]
    return bond_num


