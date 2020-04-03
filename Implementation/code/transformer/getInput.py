import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
d_n = 1000

'''
type=1: (13,k+17) 800-->(1,k)
type=2: (13,k+17) 800-->(13,k)
type=3: (13+k,17) 800-->(k,8)
type=4: (13+k,17) topk
'''
def GetInput(vertex_arr, msp_arr, max_atoms, atoms_type, k=30, type=1):
    msp_arr = torch.tensor(msp_arr,dtype=torch.int64)
    new_vertex = torch.zeros((len(vertex_arr), max_atoms, max_atoms+atoms_type))  # N,13,4+13=17
    for i in range(len(vertex_arr)):
        for j in range(len(vertex_arr[i])):
            new_vertex[i, j, 0:4] = torch.tensor([1 if ii==vertex_arr[i][j] else 0 for ii in range(atoms_type)])
            new_vertex[i,j,4:] = torch.tensor([1 if ii==j else 0 for ii in range(max_atoms)])

    if type==1:
        new_inputs = torch.zeros((len(vertex_arr), max_atoms, atoms_type + max_atoms + k+1),dtype=torch.float32)  # [N,13,17+k=47]
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
        new_inputs = torch.zeros((len(vertex_arr), max_atoms+k, atoms_type + max_atoms+1),
                                 dtype=torch.float32)  # [N,13+k,17]
        new_inputs[:, 0:13,:-1] = new_vertex
        for i in range(k):
            embedding = nn.Embedding(d_n, max_atoms+atoms_type, padding_idx=0)
            new_msp = embedding(msp_arr)  # (N,1,800,17)
            new_msp = torch.sum(new_msp, dim=1)  # (N,17)
            new_msp = F.normalize(new_msp, p=2, dim=1) # normalized msp (N,1,17)
            new_inputs[:,max_atoms+i, :-1] = new_msp
    elif type==4:
        new_inputs = torch.zeros((len(vertex_arr), max_atoms + k, atoms_type + max_atoms+1),
                                 dtype=torch.float32)  # [N,13+k,17]
        new_inputs[:, 0:13, :-1] = new_vertex
        new_msp = torch.zeros((len(msp_arr),k),dtype=torch.int64) #[N,k]
        for i in range(len(msp_arr)):
            peaks = []
            for j in range(1, 800 - 1):
                if msp_arr[i,j - 1] < msp_arr[i,j] and msp_arr[i,j + 1] < msp_arr[i,j]: peaks.append(msp_arr[i,j])
            peaks.sort(reverse=True)  # descending
            while (len(peaks) < k):
                peaks.append(0)
            new_msp[i] = torch.tensor(peaks[:k],dtype=torch.int64)
        embedding = nn.Embedding(d_n, max_atoms + atoms_type)
        new_msp = embedding(new_msp)  # (N,1,k,17)
        new_msp = F.normalize(new_msp, p=2, dim=2)  # normalized msp
        new_inputs[:, max_atoms:, :-1] = new_msp
    return new_inputs
