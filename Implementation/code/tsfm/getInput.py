import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

criterion = nn.CrossEntropyLoss()
d_n=800
atom_mass = [12,1,16,14]
atom_weight = [4,1,2,3]
atom_valence = [[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]]
max_atoms = 13
output_num = 4
negative_num = -100000000

max_edge = [4,1,2,3]
edge_max = [3,1,2,3]
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

def find(num,max_atoms):
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
           x, y = find(actions[i] // output_num)
           c = actions[i] % output_num
           if x<len(atoms[i]) and y<len(atoms[i]):
               decoder_inputs[i, actions[i] // output_num, 0] = 1
               decoder_inputs[i, actions[i] // output_num, 1] = c + 1
               decoder_inputs[i, actions[i] // output_num, 2:5] = torch.tensor(temp[c])
               decoder_inputs[i, x + msp_len, 18] = min(c + 1+decoder_inputs[i, x + msp_len, 18],decoder_inputs[i, x + msp_len, 19])
               decoder_inputs[i, y + msp_len, 18] =min(c + 1+decoder_inputs[i, x + msp_len, 18],decoder_inputs[i, x + msp_len, 19])
               decoder_inputs[i, x + msp_len, 19] =max(decoder_inputs[i, x + msp_len, 19]-c-1,0)
               decoder_inputs[i, y + msp_len, 19] =max(decoder_inputs[i, x + msp_len, 19]-c-1,0)
               decoder_embedding[i, actions[i] // output_num, 1 * per_len:2 * per_len] = embedding_binary(
                   decoder_inputs[i, actions[i] // output_num, 0])
               decoder_embedding[i, actions[i] // output_num, 1 * per_len:2 * per_len] = embedding_bond(
                   decoder_inputs[i, actions[i] // output_num, 1])
               decoder_embedding[i, actions[i] // output_num, 2 * per_len:5 * per_len] = embedding_binary(
                   decoder_inputs[i, actions[i] // output_num, 2:5]).view(-1)
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


def GetDecoderEdges(mol_adj_arr,atom_vertex,max_atoms,padding_idx,type,actions=None):
    #[label presence 1(0: exist, 1:no bond), label type 3 [0 0 0], weight 1, 5
    # atom1 mass 1
    # atom2 mass 1
    length = int(max_atoms*(max_atoms-1)//2)
    l2 = 7
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
                x, y = find(actions[i] // output_num,max_atoms)
                mol_adj_arr[i, x, y] = actions[i] % output_num
                mol_adj_arr[i, y, x] = actions[i] % output_num
        idx = 0
        for ii in range(len(atom_vertex[i])-1):
            for jj in range(ii+1,len(atom_vertex[i])):
                if mol_adj_arr[i,ii,jj]==0:
                    edge_adj[i,idx,0]=1 #no label
                else:
                    edge_adj[i,idx,0]=0 #is label
                edge_adj[i,idx,1:4] = torch.tensor([1 if pp==mol_adj_arr[i,ii,jj] else 0 for pp in [1,2,3]]) #[0 1 0] label type
                edge_adj[i,idx,4] = mol_adj_arr[i, ii, jj] # weight
                #edge_adj[i, idx, 5:8] = torch.tensor(edge_type[int(atom_vertex[i][ii])][int(atom_vertex[i][jj])]) #possible bond
                edge_adj[i, idx, 5] = atom_mass[int(atom_vertex[i][ii])]
                edge_adj[i, idx, 6]=atom_mass[int(atom_vertex[i][jj])]
                idx +=1
            if padding_idx is not None:
                for jj in range(len(atom_vertex[i]), max_atoms):
                    edge_adj[i, idx] = torch.tensor([padding_idx] * l2)
                    idx += 1
        if padding_idx is not None:
            for ii in range(len(atom_vertex[i]) - 1, max_atoms - 1):
                for jj in range(ii + 1, max_atoms):
                    edge_adj[i, idx] = torch.tensor([padding_idx] * l2)
                    idx += 1
    return edge_adj, l2

def edge_mat(batch_size,edge_lists,atoms,max_atoms):
    pred_mat = torch.zeros((batch_size, max_atoms, max_atoms))
    preds = torch.Tensor(edge_lists)
    for i in range(batch_size):
        g1 = preds[:, i]  # [edges per mol] [89,55,77,...]
        for j in range(len(atoms[i])):
            if g1[j][0] == -1 and j == len(g1) - 1: break
            elif g1[j][0] == -1: continue
            x, y = find(int(g1[j][0]),max_atoms)
            pred_mat[i, x, y] = int(g1[j][1])
            pred_mat[i, y, x] = int(g1[j][1])
            if j == len(g1) - 1: break
    return pred_mat

def select_action(probs,labels,edge_lists,atoms,batch_size,max_atoms,h_num):
    flag_mat = torch.ones_like(probs)
    flag_mat[:,:,0] = 0
    #get prediction matrix 13*13
    pred_mat = edge_mat(batch_size,edge_lists,atoms,max_atoms)
    for b in range(len(atoms)): #each molecule
        for ii in range(len(edge_lists)): #each edge per molecule
            if edge_lists[ii][b][0] == -1:
                continue
            else:
                #let exist action to be impossible 1
                for o in range(1, output_num):
                    flag_mat[b, edge_lists[ii][b][0], o] = 0
                    if probs[b, edge_lists[ii][b][0], o] > 0:
                        probs[b, edge_lists[ii][b][0], o] *= negative_num
                    else:
                        probs[b, edge_lists[ii][b][0], o] *= -negative_num
    #impossible edge process
    flag = []
    for b in range(len(atoms)):
        start, end = 0, -1
        sum_left = 0
        for ii in range(len(atoms[b])): #per line
            num = max_edge[int(atoms[b][ii])]  # max sum per line
            e_max = int(edge_max[int(atoms[b][ii])])
            s = pred_mat[b, ii].sum() #current row sum
            start = end + 1
            end = start + max_atoms - ii - 2
            sum_left += (num-s)
            for p in range(max_atoms - len(atoms[b])): #non-exist atom
                for o in range(1,output_num):
                    flag_mat[b, p+1 + end - (max_atoms - len(atoms[b])), o] = 0
                    if probs[b, p+1 + end - (max_atoms - len(atoms[b])), o] > 0:
                        probs[b, p+1 + end - (max_atoms - len(atoms[b])), o] *= negative_num
                    else:
                        probs[b, p+1 + end - (max_atoms - len(atoms[b])), o] *= -negative_num
            if int(num-s) < e_max: e_max = int(num-s)
            # if num-s==0 1,2,3=0; num-s==1 2,3=0; num-s==2 3=0
            for row in range(start, end + 1):  #row
                for o in range(e_max + 1, output_num):
                    flag_mat[b, row, o] = 0
                    if probs[b, row, o] > 0:
                        probs[b, row, o] *= negative_num
                    else:
                        probs[b, row, o] *= -negative_num
            idx = ii - 1
            for col in range(ii):  # column
                for o in range(e_max + 1, output_num):
                    flag_mat[b, idx, o] = 0
                    if probs[b, idx, o] > 0:
                        probs[b, idx, o] *= negative_num
                    else:
                        probs[b, idx, o] *= -negative_num
                idx += max_atoms - col - 2
        #non-atom impossible 2
        for qq in range(end+1,len(probs[b])):
            for o in range(1,output_num):
                flag_mat[b, qq, o] = 0
                if probs[b, qq, o] > 0:
                    probs[b, qq, o] *= negative_num
                else:
                    probs[b, qq, o] *= -negative_num
        hn = h_num[b].split('H')
        nn = 0
        if len(hn)>1: nn = int(hn[1][0])
        if int(sum_left) <= nn:
            flag.append(0) #stop sampling
        else: flag.append(1)
        #sum of each atoms' valence - sum of current bond <= number of H

    log_probs = []
    rewards = []
    actions = []
    edge_list = []
    probs = F.softmax(probs.view(len(atoms), -1), dim=1)
    for i in range(batch_size):
        if flag_mat[i].sum() ==0 or flag[i]==0:
            actions.append(-1)
            edge_list.append([-1, -1])
            rewards.append(0)
            log_probs.append(0)
            continue
        m = torch.distributions.Categorical(probs[i])  # [batch, 78*output_num=234]
        while(1):
            action = m.sample()
            pos = action // output_num
            if not (flag_mat[i,pos,1] ==0 and flag_mat[i,pos,2]==0 and flag_mat[i,pos,3] == 0): break
        x, y = find(action // output_num,max_atoms)
        actions.append(int(action))
        if int(labels[i][x][y]) == int(action % output_num):
            reward = 1
        else:
            reward = 0
        rewards.append(reward)
        log_probs.append(torch.log(probs[i][action]))
        edge_list.append([int(action // output_num), int(action % output_num)])
    return log_probs, rewards, actions, edge_list

def find_permutation(atom_input,start=output_num):
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
def getGraph(E,atom_list):
    new_E = np.zeros(E.shape)
    for ii in range(len(atom_list)):  # per atom
        for jj in range(len(atom_list)):  # per atom
            new_E[ii, jj] = E[atom_list[ii], atom_list[jj]]
    return new_E

def select_loss(policy,seq_len,atom_input,labels,edge_lists):
    pred_mat = edge_mat(seq_len,edge_lists,atom_input,max_atoms)
    batch_loss = []
    #reward_episode = torch.zeros_like(policy.reward_episode)
    edge_lists = torch.Tensor(edge_lists)
    for i in range(seq_len):
        max_rewards = []
        edge = edge_lists[:,i]
        atom_lists = find_permutation(atom_input[i], start=3)  # [[1,2,3],[2,1,3]]
        tot = 0
        for xx in range(len(labels[i])):
            for yy in range(len(labels[i][xx])):
                if labels[i][xx][yy]!=0: tot+=1
        reward_episode = 0
        for indexs in range(len(atom_lists)):
            new_E = np.zeros(pred_mat[0].shape)
            for ii in range(len(atom_lists[indexs])):  # per atom
                for jj in range(len(atom_lists[indexs])):  # per atom
                    new_E[ii, jj] = pred_mat[i][atom_lists[indexs][ii], atom_lists[indexs][jj]]
            for j in range(2,len(edge)):
                if edge[j][0] != -1:
                    x, y = find(int(edge[j][0]),max_atoms)
                    if new_E[x, y] == labels[i][x][y]:
                        reward_episode += 1.0 #= 1.0 + policy.gamma * reward_episode
            max_rewards.append(reward_episode/tot)
        log_probs = []
        for pol in policy.policy_history:
            for bb in range(len(pol)):
                if i==bb:
                    log_probs.append(pol[bb])
                    break
        idx = 0
        max_reward = max(max_rewards)
        #max_reward = min(max_rewards)
        policy_gradient = []
        for log_prob in log_probs:
            if log_prob!=0:
                policy_gradient.append(-log_prob * max_reward)
                idx += 1
                if idx == len(atom_input[i]): break
        # for one molecule one permutation
        if len(policy_gradient)>0:
            loss = torch.stack(policy_gradient).mean()
            batch_loss.append(loss)
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = []
    policy.reward_episode = []
    if len(batch_loss)>0:
        min_loss = torch.stack(batch_loss).sum()
        policy.loss_history.append(min_loss.item())
        return min_loss,1
    else:
        return 0,0