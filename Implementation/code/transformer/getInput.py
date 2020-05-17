import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
d_n=800
atom_mass = [12,1,16,14]
atom_weight = [4,1,2,3]
atom_valence = [[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]]
max_atoms = 13

max_edge = [4,1,2,3]
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
type=8: (13 atoms:atom_mass + atom_pos; k: x+y values
'''
def GetInput(vertex_arr, msp_arr, max_atoms, atoms_type, d_model=32, k=30, type=1):
    atom_num = torch.zeros((len(vertex_arr)),dtype=torch.int64)
    if type not in (7,8):
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
        new_inputs[:, max_atoms:, :] = new_msp
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
    elif type == 8:
        new_inputs = torch.zeros((len(vertex_arr), max_atoms + k, d_model * (max_atoms + atoms_type)),
                                 dtype=torch.float32)  # [N,13+k,304]
        new_inputs[:, 0:13, d_model * (max_atoms + atoms_type) - max_atoms:] = new_vertex[:, :, 1:]
        embedding1 = nn.Embedding(atoms_type, d_model * (max_atoms + atoms_type) - max_atoms)
        embed_vertex = embedding1(new_vertex[:, :, 1]).view(len(vertex_arr), max_atoms,
                                                            -1)  # [225, 13, d_model*10-max_atoms]
        new_inputs[:, 0:13, :d_model * (max_atoms + atoms_type) - max_atoms] = F.normalize(embed_vertex, dim=2)
        new_msp = torch.zeros((len(msp_arr), k), dtype=torch.int64)  # [N,k]
        msp_y = torch.zeros((len(msp_arr), k), dtype=torch.int64)  # [N,k]
        for i in range(len(msp_arr)):
            peaks = []
            y_value = []
            for j in range(1, 800 - 1):
                if msp_arr[i, j - 1] < msp_arr[i, j] and msp_arr[i, j + 1] < msp_arr[i, j]:
                    peaks.append(j)
                    y_value.append(msp_arr[i,j])
            while (len(peaks) < k):
                peaks.append(0)
                y_value.append(0)
            new_msp[i] = torch.tensor(peaks[:k], dtype=torch.int64)
            msp_y[i] =  torch.tensor(y_value[:k], dtype=torch.int64)
        embedding2 = nn.Embedding(800, d_model * (max_atoms + atoms_type) - max_atoms)
        embedding3 = nn.Embedding(1000, max_atoms)
        new_msp = embedding2(new_msp)  # (N,1,k,17)
        new_msp = F.normalize(new_msp, p=2, dim=2)  # normalized msp
        new_inputs[:, max_atoms:, :d_model * (max_atoms + atoms_type) - max_atoms] = new_msp
        msp_y = embedding3(msp_y)
        msp_y = F.normalize(msp_y, p=2, dim=2)  # normalized msp
        new_inputs[:, max_atoms:, d_model * (max_atoms + atoms_type) - max_atoms:] = torch.zeros((msp_y),
                                                                                                 dtype=torch.float32)

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

def GetEdge(mol_adj_arr,max_atoms,type):
    #[label presence 1, weight 1, label type 3, position 13] 18 0 0
    length = int(max_atoms*(max_atoms-1)//2)
    if type=="input":
        edge_adj = torch.zeros((len(mol_adj_arr),length,20), dtype=torch.int64)
    else:
        edge_adj = torch.zeros((len(mol_adj_arr), length, 20), dtype=torch.float32)
    for i in range(len(mol_adj_arr)):
        idx = 0
        for ii in range(max_atoms-1):
            for jj in range(ii+1,max_atoms):
                if mol_adj_arr[i,ii,jj]==0:
                    edge_adj[i,idx,0]=0
                else:
                    edge_adj[i,idx,0]=1
                edge_adj[i,idx,1] = mol_adj_arr[i, ii, jj]
                edge_adj[i,idx,2:5] = torch.tensor([1 if pp==mol_adj_arr[i,ii,jj] else 0 for pp in [1,2,3]])
                edge_adj[i,idx, 5:18] = torch.tensor([1 if pp == ii or pp==jj else 0 for pp in range(max_atoms)])
                idx +=1
    return edge_adj


def GetMSEmbedding(msp_arr,d_model,k=30):
    new_inputs = torch.zeros((len(msp_arr), k, d_model),dtype=torch.float32)  # [N,13+k,304]
    new_msp = torch.zeros((len(msp_arr), k), dtype=torch.int64)  # [N,k]
    msp_y = torch.zeros((len(msp_arr), k), dtype=torch.int64)  # [N,k]
    for i in range(len(msp_arr)):
        peaks = []
        y_value = []
        for j in range(1, 800 - 1):
            if msp_arr[i, j - 1] < msp_arr[i, j] and msp_arr[i, j + 1] < msp_arr[i, j]:
                peaks.append(j)
                y_value.append(msp_arr[i, j])
        while (len(peaks) < k):
            peaks.append(0)
            y_value.append(0)
        new_msp[i] = torch.tensor(peaks[:k], dtype=torch.int64)
        msp_y[i] = torch.tensor(y_value[:k], dtype=torch.int64)
    embedding2 = nn.Embedding(800, d_model//2)
    embedding3 = nn.Embedding(1000, d_model//2)
    new_msp = embedding2(new_msp)  # (N,1,k,17)
    new_msp = F.normalize(new_msp, p=2, dim=2)  # normalized msp
    new_inputs[:, :, :d_model//2] = new_msp
    msp_y = embedding3(msp_y)
    msp_y = F.normalize(msp_y, p=2, dim=2)  # normalized msp
    new_inputs[:, :, d_model//2:] = msp_y
    return new_inputs

def find(num,max_atoms=13):
    ls = []
    idx = 0
    for i in range(max_atoms - 1):
        for j in range(i + 1, max_atoms):
            ls.append([i, j])
            idx += 1
    return ls[num][0], ls[num][1]

def GetDecoderInput(edges,atoms,d_model,actions=None):
    """
    :param edges: batch, 78*[edge type 3] [0 0 1]
    :param atoms: num, 13*[atom type C/O/N...] [2]
    :return:
    [atoms_out edges_out]: num, 13*[potision 13, mass 1, valence 4, current edge 1, remaining ege 1] 20
    ,78*[label presence 1, weight 1, label type 3, position 13] 18
    """
    per_len = 6
    size0 = edges.size(0)
    embedding_binary = nn.Embedding(2,per_len)
    embedding_mass = nn.Embedding(3,per_len)
    embedding_bond = nn.Embedding(5,per_len)
    msp_len = int(max_atoms*(max_atoms-1)//2)
    decoder_inputs = torch.zeros((len(edges), msp_len + max_atoms,20), dtype=torch.int64)  # [N,13+k,304]
    decoder_embedding = torch.zeros((len(edges), msp_len + max_atoms, d_model), dtype=torch.float32)  # [N,13+k,304]
    decoder_inputs[:,:msp_len]=edges
    decoder_embedding[:, :msp_len] = embedding_bond(decoder_inputs[:,:msp_len]).view(size0,msp_len,-1)
    temp = [[1,0,0],[0,1,0],[0,0,1]]
    for i in range(len(atoms)):
        for j in range(len(atoms[i])):
            decoder_inputs[i, j+msp_len,0] =atom_mass[int(atoms[i][j])] #mass
            decoder_inputs[i, j+msp_len, 1:5] =torch.tensor(atom_valence[int(atoms[i][j])])
            decoder_inputs[i, j+msp_len, 5:18] = torch.tensor([1 if j==p else 0 for p in range(max_atoms)])
            decoder_inputs[i, j+msp_len, 19] = atom_weight[int(atoms[i][j])]
            decoder_embedding[i, j+msp_len, 0*per_len:1*per_len] = embedding_mass(decoder_inputs[i, j,0])
            decoder_embedding[i, j+msp_len, 1*per_len:5*per_len] = embedding_binary(decoder_inputs[i, j, 1:5]).view(-1)
            decoder_embedding[i, j+msp_len, 5*per_len:18*per_len] = embedding_binary(decoder_inputs[i, j, 5:18]).view(-1)
            decoder_embedding[i, j+msp_len, 18*per_len:19*per_len] = embedding_bond(decoder_inputs[i, j, 19])
        if actions is not None:
           x, y = find(actions[i] // 3)
           c = actions[i] % 3
           if x<len(atoms[i]) and y<len(atoms[i]):
               decoder_inputs[i, actions[i] // 3, 0] = 1
               decoder_inputs[i, actions[i] // 3, 1] = c + 1
               decoder_inputs[i, actions[i] // 3, 2:5] = torch.tensor(temp[c])
               decoder_inputs[i, x + msp_len, 18] = min(c + 1+decoder_inputs[i, x + msp_len, 18],decoder_inputs[i, x + msp_len, 19])
               decoder_inputs[i, y + msp_len, 18] =min(c + 1+decoder_inputs[i, x + msp_len, 18],decoder_inputs[i, x + msp_len, 19])
               decoder_inputs[i, x + msp_len, 19] =max(decoder_inputs[i, x + msp_len, 19]-c-1,0)
               decoder_inputs[i, y + msp_len, 19] =max(decoder_inputs[i, x + msp_len, 19]-c-1,0)
               decoder_embedding[i, actions[i] // 3, 1 * per_len:2 * per_len] = embedding_binary(
                   decoder_inputs[i, actions[i] // 3, 0])
               decoder_embedding[i, actions[i] // 3, 1 * per_len:2 * per_len] = embedding_bond(
                   decoder_inputs[i, actions[i] // 3, 1])
               decoder_embedding[i, actions[i] // 3, 2 * per_len:5 * per_len] = embedding_binary(
                   decoder_inputs[i, actions[i] // 3, 2:5]).view(-1)
               decoder_embedding[i, x + msp_len, 18 * per_len:19 * per_len] = embedding_bond(
                   decoder_inputs[i, x + msp_len, 18])
               decoder_embedding[i, y + msp_len, 18 * per_len:19 * per_len] = embedding_bond(
                   decoder_inputs[i, y + msp_len, 18])
               decoder_embedding[i, x + msp_len, 19 * per_len:20 * per_len] = embedding_bond(
                   decoder_inputs[i, x + msp_len, 19])
               decoder_embedding[i, y + msp_len, 19 * per_len:20 * per_len] = embedding_bond(
                   decoder_inputs[i, y + msp_len, 19])

    return decoder_inputs


def GetMSInput(msp_arr,max_len=30): #padding is 0 1e-4
    new_inputs = torch.zeros((len(msp_arr), max_len, 2),dtype=torch.long)  # [N,1k,2]
    for i in range(len(msp_arr)):
        idx = 0
        for j in range(1, len(msp_arr[i])-1):
            if msp_arr[i, j - 1] < msp_arr[i, j] and msp_arr[i, j + 1] < msp_arr[i, j]:
                new_inputs[i,idx,0] = j
                new_inputs[i,idx,1] = msp_arr[i, j]
                idx+=1
                if idx==max_len: return new_inputs
    return new_inputs

# def GetDecoderEdges(mol_adj_arr,atom_vertex,max_atoms,padding_idx,type,actions=None):
#     #[label presence 1(0: exist, 1:no bond), label type 3, weight 1, position 13, valence type 4(0 1 2 3),
#     # atom1 mass 1, atom1 valence 4(1 2 3 4), current sum valence 1, remain valence 1,
#     # atom2 mass 1, atom2 valence 4, current sum valence 1, remain valence 1] 36
#     length = int(max_atoms*(max_atoms-1)//2)
#     l2 = 36
#     edge_type = [[[1,1,1,1],[1,1,0,0],[1,1,1,0],[1,1,1,1]],
#                  [[1,1,0,0],[1,1,0,0],[1,1,0,0],[1,1,0,0]],
#                  [[1,1,1,0],[1,1,0,0],[1,1,1,0],[1,1,1,0]],
#                  [[1,1,1,1],[1,1,0,0],[1,1,1,0],[1,1,1,1]]] #C H O N
#     if type=="input":
#         edge_adj = torch.zeros((len(mol_adj_arr),length,l2), dtype=torch.int64)
#     else:
#         edge_adj = torch.zeros((len(mol_adj_arr), length, l2), dtype=torch.float32)
#     for i in range(len(mol_adj_arr)):
#         mol_adj_arr[i,0,1] = 2
#         mol_adj_arr[i, 0, 2] = 1
#         mol_adj_arr[i, 1, 0] =2
#         mol_adj_arr[i, 2, 0] = 1
#         if actions is not None:
#             x, y = find(actions[i] // 3)
#             mol_adj_arr[i,x,y] = actions[i] % 3 +1
#             mol_adj_arr[i, y, x] = actions[i] % 3 + 1
#         idx = 0
#         for ii in range(len(atom_vertex[i])-1):
#             for jj in range(ii+1,len(atom_vertex[i])):
#                 if mol_adj_arr[i,ii,jj]==0:
#                     edge_adj[i,idx,0]=1 #no label
#                 else:
#                     edge_adj[i,idx,0]=0 #is label
#                 edge_adj[i,idx,1:4] = torch.tensor([1 if pp==mol_adj_arr[i,ii,jj] else 0 for pp in [1,2,3]]) #[0 1 0]
#                 edge_adj[i,idx,4] = mol_adj_arr[i, ii, jj] #[2]
#                 edge_adj[i,idx, 5:18] = torch.tensor([1 if pp == ii or pp==jj else 0 for pp in range(max_atoms)]) #[0 0 1 1 0 ...]
#                 edge_adj[i, idx, 18:22] = torch.tensor(edge_type[int(atom_vertex[i][ii])][int(atom_vertex[i][jj])])
#                 edge_adj[i, idx, 22] = atom_mass[int(atom_vertex[i][ii])]
#                 edge_adj[i, idx, 23:27] = torch.tensor(atom_valence[int(atom_vertex[i][ii])])
#                 edge_adj[i, idx, 27] = min(atom_weight[int(atom_vertex[i][ii])],sum(mol_adj_arr[i][ii]))
#                 edge_adj[i, idx, 28] = max(0,atom_weight[int(atom_vertex[i][ii])] - edge_adj[i, idx, 27])
#                 edge_adj[i, idx, 29]=atom_mass[int(atom_vertex[i][jj])]
#                 edge_adj[i, idx, 30:34] = torch.tensor(atom_valence[int(atom_vertex[i][jj])])
#                 edge_adj[i, idx, 34] = min(atom_weight[int(atom_vertex[i][jj])],sum(mol_adj_arr[i][jj]))
#                 edge_adj[i, idx, 35] = max(0,atom_weight[int(atom_vertex[i][jj])] - edge_adj[i, idx, 34])
#                 idx +=1
#             for jj in range(len(atom_vertex[i]),max_atoms):
#                 edge_adj[i,idx]=torch.tensor([padding_idx] * l2)
#                 idx += 1
#         for ii in range(len(atom_vertex[i])-1,max_atoms-1):
#             for jj in range(ii + 1, max_atoms):
#                 edge_adj[i, idx] = torch.tensor([padding_idx] * l2)
#                 idx += 1
#     return edge_adj

def GetDecoderEdges(mol_adj_arr,atom_vertex,max_atoms,padding_idx,type,actions=None):
    #[label presence 1(0: exist, 1:no bond), label type 3, weight 1,valence type 3(1 2 3), 8
    # atom1 mass 1
    # atom2 mass 1
    length = int(max_atoms*(max_atoms-1)//2)
    l2 = 10
    edge_type = [[[ 1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
                 [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
                 [[1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0]],
                 [[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]]]  # C H O N
    if type=="input":
        edge_adj = torch.zeros((len(mol_adj_arr),length,l2), dtype=torch.int64)
    else:
        edge_adj = torch.zeros((len(mol_adj_arr), length, l2), dtype=torch.float32)
    for i in range(len(mol_adj_arr)):
        mol_adj_arr[i,0,1] = 2 #C=O
        mol_adj_arr[i, 0, 2] = 1 #C-O
        mol_adj_arr[i, 1, 0] =2 #C=O
        mol_adj_arr[i, 2, 0] = 1 #C-O
        if actions is not None:
            if actions[i]!=-1:
                x, y = find(actions[i] // 3)
                mol_adj_arr[i, x, y] = actions[i] % 3 + 1
                mol_adj_arr[i, y, x] = actions[i] % 3 + 1
        idx = 0
        for ii in range(len(atom_vertex[i])-1):
            for jj in range(ii+1,len(atom_vertex[i])):
                if mol_adj_arr[i,ii,jj]==0:
                    edge_adj[i,idx,0]=1 #no label
                else:
                    edge_adj[i,idx,0]=0 #is label
                edge_adj[i,idx,1:4] = torch.tensor([1 if pp==mol_adj_arr[i,ii,jj] else 0 for pp in [1,2,3]]) #[0 1 0] label type
                edge_adj[i,idx,4] = mol_adj_arr[i, ii, jj] # weight
                edge_adj[i, idx, 5:8] = torch.tensor(edge_type[int(atom_vertex[i][ii])][int(atom_vertex[i][jj])]) #possible bond
                edge_adj[i, idx, 8] = atom_mass[int(atom_vertex[i][ii])]
                edge_adj[i, idx, 9]=atom_mass[int(atom_vertex[i][jj])]
                idx +=1
            for jj in range(len(atom_vertex[i]),max_atoms):
                edge_adj[i,idx]=torch.tensor([padding_idx] * l2)
                idx += 1
        for ii in range(len(atom_vertex[i])-1,max_atoms-1):
            for jj in range(ii + 1, max_atoms):
                edge_adj[i, idx] = torch.tensor([padding_idx] * l2)
                idx += 1
    return edge_adj, l2

def edge_mat(batch_size,edge_lists,atoms):
    pred_mat = torch.zeros((batch_size, max_atoms, max_atoms))
    preds = torch.Tensor(edge_lists)
    for i in range(batch_size):
        g1 = preds[:, i]  # [edges per mol] [89,55,77,...] 13
        for j in range(len(atoms[i])):
            if g1[j][0] == -1: continue
            x, y = find(int(g1[j][0]))
            pred_mat[i, x, y] = int(g1[j][1])
            pred_mat[i, y, x] = int(g1[j][1])
            if j == len(g1) - 1: break
    return pred_mat

def select_action(probs,labels,edge_lists,atoms,batch_size,max_atoms):
    flags = torch.ones_like(probs)
    pred_mat = edge_mat(batch_size,edge_lists,atoms)
    for b in range(len(atoms)):
        for ii in range(len(atoms[b])):
            # actions
            if ii < len(edge_lists):
                if edge_lists[ii][b][0] == -1:
                    continue
                else:
                    flags[b, edge_lists[ii][b][0] * 3:edge_lists[ii][b][0] * 3 + 3] = 0
    # print(pred_mat[0])
    for b in range(len(atoms)):
        start, end = 0, -1
        for ii in range(len(atoms[b])):
            # max num
            num = max_edge[int(atoms[b][ii])]  # max num per line
            s = pred_mat[b, ii].sum()
            start = end + 1
            end = start + max_atoms - ii - 2
            flags[b, (1 + end - (max_atoms - len(atoms[b]))) * 3:(end + 1) * 3] = 0
            # if b==0:
            #     print(ii, num-s,start,end)
            # print(output[b])
            if (num - s) <= 0:
                flags[b, start * 3:(end + 1) * 3] = 0
                idx = ii - 1
                for pp in range(ii):
                    flags[b, idx * 3:idx * 3 + 3] = 0
                    idx += max_atoms - pp - 2
            elif (num - s) == 1:
                for k in range(start * 3, (end + 1) * 3):
                    if k % 3 != 0:
                        flags[b, k] = 0
                idx = ii - 1
                for pp in range(ii):
                    flags[b, idx * 3 + 1:idx * 3 + 3] = 0
                    idx += max_atoms - pp - 2
                # if b==0:
                #     print(start * 3, (end + 1) * 3)
                #     print(output[b])
            elif (num - s) == 2:
                for k in range(start * 3, (end + 1) * 3):
                    if k % 3 == 2:
                        flags[b, k] = 0
                idx = ii - 1
                for pp in range(ii):
                    flags[b, idx * 3 + 2] = 0
                    idx += max_atoms - pp - 2
                # if b==0:
                #     print(start * 3, (end + 1) * 3)
                #     print(output[b])
        flags[b, (end + 1) * 3:] = 0

        log_probs = []
        rewards = []
        actions = []
        edge_list = []
        for i in range(batch_size):
            if flags[i].sum() == 0:
                actions.append(-1)
                edge_list.append([-1, -1])
                rewards.append(0)
                log_probs.append(0)
                continue
            m = torch.distributions.Categorical(probs[i])  # [batch, 78*3=234]
            while(1):
                action = m.sample()
                if flags[i,action] !=0:break
            x, y = find(action // 3)
            actions.append(int(action))
            if int(labels[i][x][y]) == int(action % 3) + 1:
                reward = 1
            else:
                reward = 0
            rewards.append(reward)
            log_probs.append(torch.log(probs[i][action]))
            edge_list.append([int(action // 3), int(action % 3) + 1])
        return log_probs,rewards,actions,edge_list

def find_permutation(atom_input,start=3):
    def permutations(per_atom: list) -> list:
        ll = []
        new_atom = per_atom.copy()
        for i in range(len(per_atom)):
            a = per_atom[i]
            new_atom.remove(a)
            if len(new_atom) == 0:
                ll.append([a])
                return ll
            for t in permutations(new_atom):
                ll.append([a] + t)
            new_atom = per_atom.copy()
        return ll

    atom_list = []
    C = []
    O = []
    N = []
    for idx in range(start, len(atom_input)):
        if atom_input[idx] == 0:
            C.append(idx)
        elif atom_input[idx] == 2:
            O.append(idx)
        elif atom_input[idx] == 3:
            N.append(idx)
    for c in (permutations(C) or [[]]):
        for o in (permutations(O) or [[]]):
            for n in (permutations(N) or [[]]):
                temp = []
                for ss in range(start):
                    temp.append(ss)
                atom_list.append(temp+c+o+n)
    return atom_list

def select_loss(policy,seq_len,atom_input,labels,edge_lists):
    pred_mat = edge_mat(seq_len,edge_lists,atom_input)
    batch_loss = []
    reward_episode = torch.Tensor(policy.reward_episode)
    for i in range(seq_len):
        #print("reward1: ",reward_episode[:][i])
        losses = []
        edge = edge_lists[:][i]
        atom_lists = find_permutation(atom_input[i], start=3) #[[1,2,3],[2,1,3]]
        for indexs in range(len(atom_lists)):
            new_E = np.zeros(pred_mat[0].shape)
            for ii in range(len(atom_lists[indexs])): #per atom
                for jj in range(len(atom_lists[indexs])): #per atom
                    new_E[ii, jj] = pred_mat[i][atom_lists[indexs][ii], atom_lists[indexs][jj]]
            for j in range(len(edge)):
                if edge[j][0] != -1:
                    x, y = find(int(edge[j][0]))
                    if new_E[x, y] == labels[i][x][y]:reward_episode[:][i][j]=1
                    else:reward_episode[:][i][j]=0
            R = 0
            rewards = []
            l_r = len(reward_episode[:][i])
            #print("reward2: ", reward_episode[:][i])
            # Discount future rewards back to the present using gamma
            for r in range(l_r):
                R = reward_episode[:][i][l_r-r-1] + policy.gamma * R
                rewards.insert(0, R)
            # Scale rewards
            rewards = torch.FloatTensor(rewards)
            rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
            # Calculate loss
            log_probs = policy.policy_history[:][i]
            idx = 0
            policy_gradient = []
            for log_prob, Gt in zip(log_probs, rewards):
                policy_gradient.append(-log_prob * Gt)
                idx += 1
                if idx == len(atom_input[i]): break
            # for one molecule one permutation
            loss = torch.stack(policy_gradient).sum()
            losses.append(loss)
        # print("******")
        # print(losses)
        batch_loss.append(min(losses))
    min_loss = torch.stack(batch_loss).sum()
    # Save and intialize episode history counters
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = []
    policy.reward_episode = []
    policy.loss_history.append(min_loss.item())

    return min_loss
