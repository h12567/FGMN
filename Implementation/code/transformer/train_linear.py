import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import copy
from tsfm.Layers import EncoderLayer
from tsfm.Sublayers import Norm
from tsfm.Embed import Embedder, PositionalEncoder
import tsfm.getInput as getInput

# vertex_arr = np.load("../tsfm/vertex_arr_new.npy", allow_pickle=True) #1843
# mol_adj_arr = np.load("../tsfm/mol_adj_arr_sort_svd.npy", allow_pickle=True)
# labels_arr = np.load("../tsfm/labels_new.npy",allow_pickle=True)
# msp_arr = np.load("../tsfm/msp_new.npy", allow_pickle=True)
# H_num = np.load("../tsfm/h_num_new.npy", allow_pickle=True)[:,0]
vertex_arr = np.load("../tsfm/vertex_arr_sort_svd.npy", allow_pickle=True) #1843
mol_adj_arr = np.load("../tsfm/mol_adj_arr_sort_svd.npy", allow_pickle=True)
msp_arr = np.load("../tsfm/msp_arr_sort.npy", allow_pickle=True)
H_num = np.load("../tsfm/h_num.npy", allow_pickle=True)[:,0]
# vertex_arr = np.load("../transformer/vertex_arr_sort_svd.npy", allow_pickle=True) #225
# mol_adj_arr = np.load("../transformer/mol_adj_arr_sort_svd.npy", allow_pickle=True)
# msp_arr = np.load("../transformer/msp_arr_sort.npy", allow_pickle=True)
# H_num = np.load("../transformer/h_num.npy", allow_pickle=True)[:,0]
# vertex_arr = np.load("../dataset/vertex_arr_sort_svd.npy", allow_pickle=True) #225
# mol_adj_arr = np.load("../dataset/mol_adj_arr_sort_svd.npy", allow_pickle=True)
# msp_arr = np.load("../dataset/msp_arr_sort.npy", allow_pickle=True)
# H_num = np.load("../dataset/h_num.npy", allow_pickle=True)[:,0]
msp_len = 800
k=20
atom_type=2
padding_idx = 799 #must > 0 and < msp_len
dropout = 0.1
batch_size = 1
atom_mass = [12,1,16,14] #[12,1,16]
atomic_number = [[1,0,0],[0,0,0],[0,1,0],[0,0,1]] #[C, H, O, N}
bond_number = [4,1,2] #[C, H, O]
default_valence = [[1,1,1],[1,0,0],[1,1,0],[1,1,1]]
atom_pos_mass = [28,29,33,40,32+12,16*3,12+4,24+1,24+2,24+3,28+2,28+3,12*3,36+1,38,24+16,41,41,42,48,49,36+16,52,53,28*2,56,57,12*5]

edge_num = 78 #3#29*15#78 #3
d_model=256
max_atoms = 13 # 3
max_len1 = 2+edge_num+k
max_len2 = (2+1)*edge_num+k
max_len3 = (max_atoms*(max_atoms+1)+k)
max_len5 = max_atoms*4+k
max_len6 = 43 + max_atoms * 3 + k
max_len7 = 7*edge_num+k
max_len9 = 2*edge_num+k
max_len10 = max_atoms*2 + 2 + edge_num*4 + k
max_len11 = (edge_num+max_atoms)*4 + k

def countH(h_num):
    count = 0
    if "H" in h_num:
        temp = h_num.split('H')
        if len(temp[1]) == 0:
            count = 1
        else:
            for t in temp[1]:
                if t.isdigit():
                    count = count * 10 + int(t)
                else:
                    break
    return count
def getEdgeIdx(pos1,pos2=None): #not contain H
    edge_idx = 0
    for jj in range(pos1):
        edge_idx += (max_atoms - jj - 1)
    if pos2==None:
        return edge_idx-1
    else:
        edge_idx += pos2 - pos1
    return edge_idx-1
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx)#.unsqueeze(-2)
def get_pad_mask1(seq, pad_idx):
    mask =  torch.ByteTensor([[[False]*4]*seq.size(1)]*seq.size(0)) #torch.zeros((seq.size(0),seq.size(1),4))
    for b in range(seq.size(0)):
        for i in range(seq.size(1)):
            if seq[b,i]==pad_idx: mask[b,i] = torch.ByteTensor([0,1,1,1])
    return mask
def get_pad_mask2(seq, pad_idx):
    mask =  torch.ByteTensor([[[False]*4]*edge_num]*seq.size(0)) #torch.zeros((seq.size(0),seq.size(1),4))
    for b in range(seq.size(0)):
        for i in range(edge_num):
            if seq[b,i*3]==pad_idx: mask[b,i] = torch.ByteTensor([0,1,1,1])
    return mask
def get_pad_mask3(seq, pad_idx):
    mask =  torch.ByteTensor([[[False]*29]*max_atoms]*seq.size(0)) #torch.zeros((seq.size(0),seq.size(1),28))
    for b in range(seq.size(0)):
        for i in range(max_atoms):
            if seq[b,i*(max_atoms+1)]==pad_idx: mask[b,i] = torch.ByteTensor([0]+[1]*28)
    return mask
def get_pad_mask4(seq, pad_idx):
    mask =  torch.ByteTensor([[[False]*29]*max_atoms]*seq.size(0)) #torch.zeros((seq.size(0),seq.size(1),28))
    for b in range(seq.size(0)):
        for i in range(max_atoms):
            if seq[b,i,0]==pad_idx: mask[b,i] = torch.ByteTensor([0]+[1]*28)
    return mask
def get_pad_mask5(seq, pad_idx):
    mask =  torch.ByteTensor([[[False]*29]*max_atoms]*seq.size(0)) #torch.zeros((seq.size(0),seq.size(1),28))
    for b in range(seq.size(0)):
        for i in range(max_atoms):
            if seq[b,i*4]==pad_idx:# or seq[b,i*4+2]==0:
                mask[b,i] = torch.ByteTensor([0]+[1]*28)
    return mask
def get_pad_mask52(seq,pad_idx): #action_pos:[pos1,po2,...]
    mask = torch.ByteTensor([[[False] * 29] * max_atoms] * seq.size(0))  # torch.zeros((seq.size(0),seq.size(1),28))
    for b in range(seq.size(0)):
        for i in range(max_atoms):
            if seq[b, i * 4] == pad_idx or seq[b,i*4+2]==0:
                mask[b, i] = torch.ByteTensor([1] * 29)
    return mask
def get_pad_mask7(seq, pad_idx):
    mask =  torch.ByteTensor([[[False]*4]*seq.size(1)]*seq.size(0)) #torch.zeros((seq.size(0),seq.size(1),4))
    for b in range(seq.size(0)):
        for i in range(seq.size(1)):
            if seq[b,i]==pad_idx: mask[b,i] = torch.ByteTensor([0,1,1,1])
    return mask
def get_pad_mask72(seq,pad_idx): #action_pos:[pos1,po2,...]
    mask = torch.ByteTensor([[[False] * 4] * edge_num] * seq.size(0))  # torch.zeros((seq.size(0),seq.size(1),28))
    for b in range(seq.size(0)):
        for i in range(edge_num):
            if 1 in seq[b, i * 7+3:i*7+7]:
                mask[b, i] = torch.ByteTensor([1] * 4)
    return mask
def get_pad_mask10(seq):
    mask = torch.ByteTensor([[[False] * 3] * (edge_num+max_atoms)] * seq.size(0))  # torch.zeros((seq.size(0),seq.size(1),28))
    for b in range(seq.size(0)):
        idx = 0
        for i in range(max_atoms):
            for j in range(i+1, max_atoms+1):
                if seq[b, i * 2+1] == 2 or seq[b, j * 2+1] == 2: # for O like
                    mask[b,idx] = torch.ByteTensor([0,0,1])
                if seq[b, j * 2]==1 or seq[b, i * 2+1] == 1 or seq[b, j * 2+1] == 1: # only bond 1
                    mask[b,idx] = torch.ByteTensor([0,1,1])
                if seq[b, i * 2] == padding_idx or seq[b, j * 2] == padding_idx\
                        or seq[b, i * 2+1] == 0 or seq[b, j * 2+1] == 0: # no bond
                    mask[b, idx] = torch.ByteTensor([1, 1, 1])
                if j<max_atoms and 1 in seq[b, (max_atoms+1)*2+getEdgeIdx(i,j)*4:(max_atoms+1)*2+getEdgeIdx(i,j)*4+4]:
                    mask[b, idx] = torch.ByteTensor([1, 1, 1])
                idx+=1
    return mask
def get_pad_mask11(seq):
    mask = torch.ByteTensor([[[False] * 4] * (edge_num+max_atoms)] * seq.size(0))  # torch.zeros((seq.size(0),seq.size(1),28))
    for b in range(seq.size(0)):
        idx = 0
        for i in range(max_atoms):
            for j in range(i+1, max_atoms+1):
                if seq[b, idx * 4] == padding_idx or seq[b, idx * 4+2] == padding_idx: mask[b, idx] = torch.ByteTensor([0, 1, 1, 1]) # no bond (bond 0 here)
                elif seq[b, idx*4+3] == 1: mask[b,idx] = torch.ByteTensor([0,0,1,1]) #only bond 1 or bond 0
                elif seq[b, idx*4+1]==2 or seq[b, idx * 4+3] == 2: mask[b,idx] = torch.ByteTensor([0,0,0,1])
                idx+=1
    return mask
class Classify0(nn.Module): #linear
    def __init__(self,padding_idx):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len,d_model,padding_idx)
        self.ll1 = nn.Linear(d_model, edge_num)
        self.ll2 = nn.Linear(k,4)
    def forward(self,src):
        #self.mask = get_pad_mask2(src, self.padding_idx)  # [batch, edge_num,4]
        output = self.embedding(src) #[batch, k, d_model]
        output = self.ll1(output) #[batch, k, edge_num]
        output = output.permute(0, 2, 1) #[batch, edge_num, k]
        output = self.ll2(output) #[batch, edge_num, 4]
        #output = output.masked_fill(self.mask, -1e9)
        output = F.log_softmax(output,dim=2)
        return output #[batch, edge_num=3, bond=4]
class Classify1(nn.Module):
    def __init__(self,padding_idx):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len, d_model,self.padding_idx)
        self.ll = nn.Linear(max_len1*d_model,4)
    def forward(self,src):
        self.mask = get_pad_mask1(src[:,:,0],self.padding_idx)#[batch, edge_num,4]
        output = self.embedding(src) #[batch, edge_num=3, per_lin=35, d_model=512]
        output = output.view(src.size(0),edge_num, -1) #[batch, edge_num=3, per_lin=35*d_model=512]
        output = self.ll(output) #[batch, edge_num=3, bond=4]
        output = output.masked_fill(self.mask,-1e9)
        output = F.log_softmax(output,dim=2)
        return output #[batch, edge_num=3, bond=4]
class Classify2(nn.Module):
    def __init__(self,padding_idx):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len,d_model,padding_idx)
        self.ll1 = nn.Linear(d_model, edge_num)
        self.ll2 = nn.Linear(max_len2,4)
    def forward(self,src):
        self.mask = get_pad_mask2(src, self.padding_idx)  # [batch, edge_num,4]
        output = self.embedding(src) #[batch, max_len=45, d_model]
        output = self.ll1(output) #[batch, max_len=45, edge_num=3]
        output = output.permute(0, 2, 1) #[batch, edge_num, max_len = 45]
        output = self.ll2(output) #[batch, edge_num=3, bond=4]
        output = output.masked_fill(self.mask, -1e9)
        output = F.log_softmax(output,dim=2)
        return output #[batch, edge_num=3, bond=4]
class Classify3(nn.Module):
    def __init__(self,padding_idx):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len,d_model,padding_idx)
        self.ll1 = nn.Linear(d_model, max_atoms)
        self.ll2 = nn.Linear(max_len3,29)
    def forward(self,src):
        self.mask = get_pad_mask3(src, self.padding_idx)  # [batch,atom_num=13,28]
        output = self.embedding(src) #[batch, max_len3=212, d_model]
        output = self.ll1(output) #[batch, max_len3=212, atom_num=13]
        output = output.permute(0, 2, 1) #[batch, atom_num=13, max_len3 = 212]
        output = self.ll2(output) #[batch, atom_num=13, tpye=29]
        output = output.masked_fill(self.mask, -1e9)
        output = F.log_softmax(output,dim=2)
        return output #[batch, edge_num=3, bond=4]
class Classify4(nn.Module):
    def __init__(self,padding_idx):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len,d_model,padding_idx)
        self.ll1 = nn.Linear(d_model*44, 29)
        #self.ll2 = nn.Linear(128,29)
    def forward(self,src):
        self.mask = get_pad_mask4(src, self.padding_idx)  # [batch,atom_num=13,28]
        output = self.embedding(src) #[batch, max_atoms=13, 44, d_model]
        output = output.view(src.size(0),max_atoms,-1) #[batch, max_atoms=13, 44*d_model]
        output = self.ll1(output) #[batch, 13,128]
        #output = self.ll2(output) #[batch, 13, tpye=29]
        output = output.masked_fill(self.mask, -1e9)
        output = F.log_softmax(output,dim=2)
        return output #[batch, edge_num=3, bond=4]
class Classify5(nn.Module): #imitation
    def __init__(self,padding_idx):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len,d_model,padding_idx)
        self.ll1 = nn.Linear(d_model, max_atoms)
        self.ll2 = nn.Linear(max_len5,29)
    def forward(self,src):
        self.mask = get_pad_mask5(src, self.padding_idx)  # [batch,atom_num=13,28]
        output = self.embedding(src) #[batch, max_len5=k+13*4=72, d_model]
        output = self.ll1(output) #[batch, max_len5=k+13*4=72, atom_num=13]
        output = output.permute(0, 2, 1) #[batch, atom_num=13, max_len5]
        output = self.ll2(output) #[batch, atom_num=13, tpye=29]
        output = output.masked_fill(self.mask, -1e9)
        #output = F.log_softmax(output,dim=2)
        return output,get_pad_mask52(src,self.padding_idx) #[batch, edge_num=3, bond=4]
class Classify6(nn.Module): #imitation
    def __init__(self,padding_idx):
        super().__init__()
        #self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len,d_model,padding_idx)
        self.ll = nn.Linear(max_len6*d_model, 4)
    def forward(self,src):
        #self.mask = get_pad_mask6(src, self.padding_idx)  # [batch,atom_num=13,28]
        output = self.embedding(src) #[batch, len6, d_model]
        output = output.view(src.size(0),-1)
        output = self.ll(output) #[batch, len6-->28]
        #output = torch.sigmoid(output)
        #output = output.masked_fill(self.mask, -1e9)
        return output #[batch, edge_num=3, bond=4]
class Classify7(nn.Module):
    def __init__(self,padding_idx):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len, d_model,self.padding_idx)
        self.ll1 = nn.Linear(d_model, edge_num)
        self.ll2 = nn.Linear(max_len7, 4)
    def forward(self,src):
        mask = get_pad_mask72(src,self.padding_idx)#[batch, edge_num,4]
        output = self.embedding(src) #[batch, len7, d_model]
        output = self.ll1(output) #[batch, len7, edge_num]
        output = output.permute(0,2,1) #[batch, edge_num, len7]
        output = self.ll2(output) #[batch, edge_num, 4]
        #output = output.masked_fill(self.mask,-1e9)
        #output = F.log_softmax(output,dim=2)
        return output,mask #[batch, edge_num=3, bond=4]\
class Classify8(nn.Module): #transformer with linear using ms only
    def __init__(self,padding_idx):
        super().__init__()
        heads=4
        self.N=3
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len, d_model,self.padding_idx)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), self.N)
        self.norm = Norm(d_model)
        self.ll1 = nn.Linear(d_model,edge_num)
        self.ll2 = nn.Linear(k,4)
    def forward(self,src): #[batch, atom_num=13, per_lin]
        mask = get_pad_mask(src,self.padding_idx).view(src.size(0),-1).unsqueeze(-1)
        output = self.embedding(src) #[batch, k, d_model=512]
        for i in range(self.N):
            output = self.layers[i](output, mask)
        output = self.ll1(output) #[batch, k, edge_num]
        output = output.permute(0, 2, 1) #[batch, edge_num, k]
        output = self.ll2(output) #[batch, edge_num, 4]
        #output = output.masked_fill(self.mask,-1e9)
        output = F.log_softmax(output,dim=2)
        return output #[batch, edge_num=3, bond=4]
class Classify9(nn.Module): #transformer with edge as well
    def __init__(self,padding_idx):
        super().__init__()
        heads=4
        self.N=1
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len, d_model,self.padding_idx)
        self.pe = PositionalEncoder(d_model=d_model, max_seq_len=edge_num*2, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), self.N)
        self.norm = Norm(d_model)
        self.ll1 = nn.Linear(d_model,edge_num)
        self.ll2 = nn.Linear(max_len2,4)
    def forward(self,src): #[batch, atom_num=13, per_lin]
        mask = get_pad_mask(src,self.padding_idx).view(src.size(0),-1).unsqueeze(-1)
        output = self.embedding(src) #[batch, k, d_model=512]
        output = torch.cat((self.pe(output[:,:2*edge_num]),output[:,2*edge_num:]),dim=1)
        for i in range(self.N):
            output = self.layers[i](output, mask)
        output = self.ll1(output) #[batch, max_len2, edge_num]
        output = output.permute(0, 2, 1) #[batch, edge_num, max_len2]
        output = self.ll2(output) #[batch, edge_num, 4]
        #output = output.masked_fill(self.mask,-1e9)
        output = F.softmax(output,dim=-1)
        return output #[batch, edge_num=3, bond=4]
class Classify10(nn.Module): #transformer imitation
    def __init__(self,padding_idx):
        super().__init__()
        heads=4
        self.N=1
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len, d_model,self.padding_idx)
        self.pe = PositionalEncoder(d_model=d_model, max_seq_len=max_len10-k, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), self.N)
        self.norm = Norm(d_model)
        self.ll1 = nn.Linear(d_model*8,3)
    def forward(self,src): #[batch, atoms, edges, peaks]
        mask = get_pad_mask(src,self.padding_idx).view(src.size(0),-1).unsqueeze(-1)
        output_mask = get_pad_mask10(src)
        output = self.embedding(src) #[batch, k, d_model=512]
        output = torch.cat((self.pe(output[:,:max_atoms*2 + 2 + edge_num*4]),output[:,max_atoms*2 + 2 + edge_num*4:]),dim=1)
        for i in range(self.N):
            output = self.layers[i](output, mask)
        new_output = torch.zeros((src.size(0),edge_num+max_atoms,8,d_model))
        for b in range(len(src)):
            idx = 0
            for i in range(max_atoms):
                for j in range(i+1, max_atoms+1):
                    new_output[b, idx, :2] = output[b, i * 2:i * 2 + 2]
                    new_output[b, idx, 2:4] = output[b, j * 2:j * 2 + 2]
                    if j==max_atoms:
                        new_output[b,idx,4:8] = 0
                    else:
                        new_output[b, idx, 4:8] = \
                            output[b, max_atoms * 2 + 2 + getEdgeIdx(i,j) * 2: max_atoms * 2 + 2 + getEdgeIdx(i, j) * 2 + 4]
                    idx+=1
                idx+=1
        output = self.ll1(new_output.view(new_output.size(0),new_output.size(1),-1)) #[batch, edge_num+max_atoms, 3]
        output = output.masked_fill(output_mask,-1e9)
        #output = F.softmax(output.view(src.size(0),-1),dim=-1)
        return output #[batch, edge_num, bond=3]
class Classify11(nn.Module): #transformer imitation
    def __init__(self,padding_idx):
        super().__init__()
        heads=4
        self.N=1
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len, d_model,self.padding_idx)
        self.pe = PositionalEncoder(d_model=d_model, max_seq_len=max_len10-k, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), self.N)
        self.norm = Norm(d_model)
        self.ll1 = nn.Linear(d_model*4,4)
    def forward(self,src): #[batch, atoms, edges, peaks]
        mask = get_pad_mask(src,self.padding_idx).view(src.size(0),-1).unsqueeze(-1)
        output_mask = get_pad_mask11(src)
        output = self.embedding(src) #[batch, k, d_model=512]
        output = torch.cat((self.pe(output[:,:(max_atoms+ edge_num)*4]),output[:,(max_atoms+ edge_num)*4:]),dim=1)
        for i in range(self.N):
            output = self.layers[i](output, mask)
        # new_output = torch.zeros((src.size(0), edge_num + max_atoms, 4+k, d_model))
        # for b in range(len(src)):
        #     idx = 0
        #     for i in range(max_atoms):
        #         for j in range(i+1, max_atoms+1):
        #             new_output[b, idx, :4] = output[b, idx*4:idx*4+4]
        #             new_output[b, idx, 4:] = output[b, max_len11-k:]
        #             idx+=1
        #         idx+=1
        # output = self.ll1(new_output.view(new_output.size(0),new_output.size(1),-1)) #[batch, edge_num+max_atoms, 3]
        output = output[:,:max_len11-k].view(src.size(0),(edge_num+max_atoms),-1) #[batch, edge_num+max_atoms, 4*d_model]
        output = self.ll1(output) #[batch, edge_num+max_atoms, 4]
        output = output.masked_fill(output_mask,-1e9)
        output = F.softmax(output,dim=-1) #[batch, edge_num+max_atoms, 4]
        return output

def getInput0(vertex, msp): #linear
    #[msp1[x], ..., mspk[y]] 30
    src = torch.LongTensor([[padding_idx]* k]*len(vertex))#[batch, k]
    for b in range(len(vertex)): #batch
        idx = 0
        for j in range(1, len(msp[b]) - 1):
            if msp[b][j - 1] < msp[b][j] and msp[b][j + 1] < msp[b][j]:
                src[b, idx] = j
                idx += 1
                if idx == k: break
    return src
def getInput1(vertex, msp):
    #[A1, A2, 1 0 0, msp1[x], ..., mspk[y]] 35
    #src = torch.zeros((len(vertex),edge_num,per_len),dtype=torch.long) #[edge_num=3,per_len=35]
    src = torch.LongTensor([[[padding_idx]* max_len1]*edge_num]*len(vertex))#[batch, edge_num, per_len]
    for b in range(len(vertex)): #batch
        idx = 0
        for i in range(max_atoms):
            for j in range(i+1, max_atoms):
                if i < len(vertex[b]) and j < len(vertex[b]):
                    src[b, idx, 0] = atom_mass[int(vertex[b][i])]
                    src[b, idx, 1] = atom_mass[int(vertex[b][j])]
                    for p in range(edge_num):
                        src[b, idx, 2 + p] = int(p == i or p == j)
                    idx2 = 2 + edge_num
                    for j in range(1, len(msp[b]) - 1):
                        if msp[b][j - 1] < msp[b][j] and msp[b][j + 1] < msp[b][j]:
                            src[b, idx, idx2] = j
                            idx2 += 1
                            if idx2 == max_len1: break
                idx += 1
    return src
def getInput2(vertex, msp):
    #[A1-weight, A2-weight, 1 0 0,A1, A3,0 1 0 A2, A3,0 0 1 msp1[x], ...,mspk[x]] 45
    #src = torch.zeros((len(vertex),max_len),dtype=torch.long) #[batch, 45]
    src = torch.LongTensor([[padding_idx]* max_len2]*len(vertex)) #[batch, 45]
    for b in range(len(vertex)): #batc
        idx1 = 0
        for i in range(max_atoms):
            for ii in range(i+1,max_atoms):
                if i < len(vertex[b]) and ii < len(vertex[b]):
                    src[b, idx1 * 3], src[b, idx1 * 3 + 1] = atom_mass[int(vertex[b][i])], atom_mass[int(vertex[b][ii])]
                    src[b, idx1 * 3 + 2] = idx1 + 1
                idx1+=1
        idx = (1+2)*edge_num
        for j in range(1, len(msp[b]) - 1):
            if msp[b][j - 1] < msp[b][j] and msp[b][j + 1] < msp[b][j]:
                src[b,idx] = j
                idx += 1
                if idx == max_len2: break
    return src
def getInput3(vertex, msp):
    #[atom1-mass(12), pos(1,0,0,...), atom2-mass(16), pos(0,1,0,...), peak1, ..., peak30] 14*13+30
    #max_len3 = (max_atoms*(max_atoms+1)+k)
    src = torch.LongTensor([[padding_idx]*max_len3]*len(vertex)) #[batch, 14*13+30]
    for b in range(len(vertex)): #batch
        for i in range(len(vertex[b])):
            src[b,i*(max_atoms+1)] = atom_mass[int(vertex[b][i])]
            src[b,i*(max_atoms+1)+1:(i+1)*(max_atoms+1)] = torch.LongTensor([int(ii==i) for ii in range(max_atoms)])
        idx = max_atoms*(max_atoms+1)
        for j in range(1, len(msp[b]) - 1):
            if msp[b][j - 1] < msp[b][j] and msp[b][j + 1] < msp[b][j]:
                src[b, idx] = j
                idx += 1
                if idx == max_len3: break
    return src
def getInput4(vertex, msp):
    #[atom1-mass(12), pos(1,0,0,...) peak1, ..., peak30] 14+30=44
    #[atom2-mass(16), pos(0,1,0,...), peak1,.., peak30]
    max_len4 = max_atoms+1+k
    src = torch.LongTensor([[[padding_idx]*max_len4]*max_atoms]*len(vertex)) #[batch, 13, 14+30]
    for b in range(len(vertex)): #batch
        for i in range(len(vertex[b])):
            src[b,i,0] = atom_mass[int(vertex[b][i])]
            src[b,i,1:(max_atoms+1)] = torch.LongTensor([int(ii==i) for ii in range(max_atoms)])
            idx = max_atoms+1
            for j in range(1, len(msp[b]) - 1):
                if msp[b][j - 1] < msp[b][j] and msp[b][j + 1] < msp[b][j]:
                    src[b, i,idx] = j
                    idx += 1
                    if idx == max_len4: break
    return src
def getInput5(vertex, msp,actions): #imitation #actions = {atom_pos:type,...}
    #[atom1-mass(12), pos(0), is_empty(1), type(0),
    # atom2-mass(16), pos(1), is_emtpy(1), type(0),peak1, ..., peak30] 4*13+30
    #max_len5 = max_atoms*4+k
    src = torch.LongTensor([[padding_idx]*max_len5]*len(vertex)) #[batch, 4*13+30]
    for b in range(len(vertex)): #batch
        for i in range(len(vertex[b])):
            src[b,i*4] = atom_mass[int(vertex[b][i])]
            src[b, i * 4+1] = i #pos
            src[b, i * 4 + 2] = 1  # is empty
            src[b, i * 4 + 3] = 0
            if actions!={} and i in actions.keys():
                src[b, i * 4 + 2] = 0
                src[b, i * 4 + 3] = actions[i]
        idx = max_atoms*4
        for j in range(1, len(msp[b]) - 1):
            if msp[b][j - 1] < msp[b][j] and msp[b][j + 1] < msp[b][j]:
                src[b, idx] = j
                idx += 1
                if idx == max_len5: break
    return src
def getInput6(vertex, msp,h_num,atom_pos): #imitation #actions = {atom_pos:reward,...} h_num=[10,2,...]
    #[O,O,C,...,C, H,H,..., 0(mass),0(reward),...., peak1,...peak30] 43+13*2+30
    src = torch.LongTensor([[padding_idx]*max_len6]*len(vertex)) #[batch,43+max_atoms*2+k]
    for b in range(len(vertex)): #batch
        for i in range(len(vertex[b])):
            src[b,i] = atom_mass[int(vertex[b][i])]
        for i in range(len(vertex[b]),max_len6-k-max_atoms*3):
            count =countH(h_num[b])
            src[b,i:i+count] = torch.LongTensor([1]*count)
        src[b,max_len6-k-max_atoms*3:max_len6-k] = 0
        for j in range(len(vertex[b])):
            if j<len(atom_pos):
                src[b,max_len6-k-max_atoms*3+j*3]=atom_pos_mass[atom_pos[j]]
                src[b, max_len6 - k - max_atoms * 3 + j * 3+1] = atom_pos[j]
                src[b, max_len6 - k - max_atoms * 3 + j * 3+2] = 0
            else:
                src[b, max_len6 - k - max_atoms * 3 + j * 3 + 0] = 0
                src[b, max_len6 - k - max_atoms * 3 + j * 3 + 1] = 0
                src[b, max_len6 - k - max_atoms * 3 + j * 3 + 2] = 1
        idx = max_len6-k
        for j in range(1, len(msp[b]) - 1):
            if msp[b][j - 1] < msp[b][j] and msp[b][j + 1] < msp[b][j]:
                src[b, idx] = j
                idx += 1
                if idx == max_len6: break
    return src
def getInput7(vertex, msp,actions): #imitation #actions = {atom_pos:type,...}
    #[atom1-mass(12), atom2-mass, pos(0),type(0,0,0,0), 7*78
    # peak1, ..., peak30] 30
    src = torch.LongTensor([[padding_idx]*max_len7]*len(vertex)) #[batch, 4*13+30]
    for b in range(len(vertex)): #batch
        idx = 0
        for i in range(len(vertex[b])):
            for j in range(i+1,len(vertex[b])):
                src[b, idx * 7] = atom_mass[int(vertex[b][i])]
                src[b, idx * 7+1] = atom_mass[int(vertex[b][j])]
                src[b, idx * 7+2] = idx
                src[b, idx * 7 + 3:idx*7+7] = 0
            if actions!={} and idx in actions.keys():
                type = actions[idx]
                src[b, idx * 7 + 3:idx * 7+7]=torch.Tensor([int(t==type) for t in range(4)])
            idx+=1
        idx = max_len7-k
        for j in range(1, len(msp[b]) - 1):
            if msp[b][j - 1] < msp[b][j] and msp[b][j + 1] < msp[b][j]:
                src[b, idx] = j
                idx += 1
                if idx == max_len7: break
    return src
def getInput9(vertex, msp): #transformer with edges
    #[A1-weight, A2-weight, 1 0 0,A1, A3,0 1 0 A2, A3,0 0 1 msp1[x], ...,mspk[x]] 45
    #src = torch.zeros((len(vertex),max_len),dtype=torch.long) #[batch, 45]
    src = torch.LongTensor([[padding_idx]* max_len2]*len(vertex)) #[batch, 45]
    for b in range(len(vertex)): #batc
        idx1 = 0
        for i in range(max_atoms):
            for ii in range(i+1,max_atoms):
                if i < len(vertex[b]) and ii < len(vertex[b]):
                    src[b, idx1 * 3], src[b, idx1 * 3 + 1] = atom_mass[int(vertex[b][i])], atom_mass[int(vertex[b][ii])]
                idx1+=1
        idx = 2*edge_num
        for j in range(1, len(msp[b]) - 1):
            if msp[b][j - 1] < msp[b][j] and msp[b][j + 1] < msp[b][j]:
                src[b,idx] = j
                idx += 1
                if idx == max_len2: break
    return src
def getInput10(vertex, msp,h_num,actions): #imitation learning; actions = [ [pos1, pos2, type] ]
    #[Atom1 (O:16) remaining bonds(2), Atom2,..., H(1),No(e.g.10), edge1(0,0,0,0),..., peak1, ..., peak30]
    #max_atoms*2 + 2 + edge_num*4 + k
    src = torch.LongTensor([[padding_idx]* max_len10]*len(vertex)) #[batch, len]
    for b in range(len(vertex)): #batch
        for i in range(max_atoms):
            if i < len(vertex[b]):
                src[b,i*2] = atom_mass[int(vertex[b][i])] #Atom1 (O:16)
                src[b, i * 2+1] = bond_number[int(vertex[b][i])] #remaining bonds(2)
        src[b, max_atoms*2] = 1 #atom H
        src[b, max_atoms * 2+1] = countH(h_num[b]) #number of H
        src[b,max_atoms * 2+2:max_atoms * 2+2+edge_num*4] = 0 #initially can add bond
        idx = max_atoms * 2 + 2 + edge_num * 4
        for j in range(1, len(msp[b]) - 1): #mass spectrum peaks
            if msp[b][j - 1] < msp[b][j] and msp[b][j + 1] < msp[b][j]:
                src[b, idx] = j
                idx += 1
                if idx == max_len10: break
        #actions
        for i in range(len(actions)):
            pos1,pos2,type = actions[i][0],actions[i][1],actions[i][2]
            src[b,pos1*2+1] = max(0,src[b,pos1*2+1]-type) #update atom node1
            src[b, pos2 * 2 + 1] = max(0, src[b,pos2*2+1]-type) #update atom node2
            if pos2==max_atoms: #if H
                continue
            edge_idx = max_atoms * 2+2+getEdgeIdx(pos1,pos2)*4
            src[b,edge_idx:edge_idx+4] = torch.LongTensor([int(ii==type) for ii in range(4)])
        for i in range(max_atoms):
            if src[b, i * 2 + 1] == 0: # cannot add any bond of atom
                for jj in range(i + 1, max_atoms):
                    idx_temp = (max_atoms + 1) * 2 + getEdgeIdx(i, jj) * 4
                    if 1 not in src[b, idx_temp:idx_temp + 4]:
                        src[b, idx_temp:idx_temp + 4] = torch.LongTensor([1, 0, 0, 0])  # bond 0 (no bond can be added)
                for kk in range(i):
                    idx_temp = (max_atoms + 1) * 2 + getEdgeIdx(kk, i) * 4
                    src[b, idx_temp:idx_temp + 4] = torch.LongTensor([1, 0, 0, 0])  # bond 0 (no bond can be added)
        #check whether stop
        ff = 1
        for i in range(max_atoms):
            if src[b,i*2] == padding_idx:continue
            elif src[b,i*2+1]>0: ff *= 0 #can add
    return src, 1-ff
def getInput11(vertex, msp): #imitation learning;
    #[edge1: Atom1 (O:16) remaining bonds(2), Atom2 (C:12), remaining bonds(4),
    # edge2: ...
    # peak1, ..., peak30] edge_num*4 + k
    src = torch.LongTensor([[padding_idx]* max_len11]*len(vertex)) #[batch, len]
    for b in range(len(vertex)): #batch
        idx_edge = 0
        for i in range(max_atoms):
            if i<len(vertex[b]):
                for j in range(i+1,max_atoms):
                    if j < len(vertex[b]):
                        src[b, idx_edge * 4] = atom_mass[int(vertex[b][i])]  # Atom1 (O:16)
                        src[b, idx_edge * 4 + 1] = bond_number[int(vertex[b][i])]  # remaining bonds(2)
                        src[b, idx_edge * 4 + 2] = atom_mass[int(vertex[b][j])]  # Atom2 (C:12)
                        src[b, idx_edge * 4 + 3] = bond_number[int(vertex[b][j])]  # remaining bonds(4)
                    idx_edge +=1
                src[b, max_atoms * 4] = atom_mass[int(vertex[b][i])]  # Atom1 (O:16)
                src[b, max_atoms * 4 + 1] = bond_number[int(vertex[b][i])]  # remaining bonds(2)
                src[b, max_atoms * 4 + 2] = 1  # Atom2 (H:1)
                src[b, max_atoms * 4 + 3] = src[b, max_atoms * 4 + 1]  # remaining bonds(1)
            else:
                for j in range(i + 1, max_atoms): idx_edge +=1

        idx = (edge_num+max_atoms)*4
        for j in range(1, len(msp[b]) - 1): #mass spectrum peaks
            if msp[b][j - 1] < msp[b][j] and msp[b][j + 1] < msp[b][j]:
                src[b, idx] = j
                idx += 1
                if idx == max_len11: break
    return src

def getLabel(mol_arr,vertex):
    #label = torch.zeros((len(mol_arr),edge_num),dtype=torch.long) #[batch, edge_num]
    label = torch.LongTensor([[padding_idx]*edge_num]*len(mol_arr))
    for b in range(len(mol_arr)): #batch
        idx = 0
        for i in range(len(mol_arr[b])):
            for j in range(i + 1, len(mol_arr[b])):
                if i < len(vertex[b]) and j < len(vertex[b]):
                    label[b, idx] = mol_arr[b][i][j]
                idx += 1
    return label
def getLabel_H(mol_arr,vertex):
    #label = torch.zeros((len(mol_arr),edge_num),dtype=torch.long) #[batch, edge_num]
    label = torch.LongTensor([[padding_idx]*(edge_num+max_atoms)]*len(mol_arr))
    for b in range(len(mol_arr)): #batch
        idx = 0
        for i in range(len(mol_arr[b])):
            for j in range(i + 1, len(mol_arr[b])):
                if i < len(vertex[b]) and j < len(vertex[b]):
                    label[b, idx] = mol_arr[b][i][j]
                idx += 1
    return label
def getLabel3(label_arr):
    #label = torch.zeros((len(mol_arr),edge_num),dtype=torch.long) #[batch, edge_num]
    label = torch.LongTensor([[padding_idx]*max_atoms]*len(label_arr))
    for b in range(len(label_arr)): #batch
        for i in range(len(label_arr[b])):
            label[b,i] = int(label_arr[b][i]+1)
    return label
def getLabel5(label_arr,action_pos): #actions = [atom_pos,...]
    #label = torch.zeros((len(mol_arr),edge_num),dtype=torch.long) #[batch, edge_num]
    label = torch.LongTensor([[padding_idx]*max_atoms]*len(label_arr))
    for b in range(len(label_arr)): #batch
        for i in range(len(label_arr[b])):
            label[b,i] = int(label_arr[b][i])
            if len(action_pos)>0 and i in action_pos:
                label[b, i] = padding_idx
    return label
def getLabel6(label_arr,atom_pos): #actions = [atom_pos,...]
    label = torch.Tensor([[0] * 4] * len(label_arr))
    count = torch.Tensor([[0] * 4] * len(label_arr))
    for b in range(len(label_arr)): #batch
        ll = len(label_arr[b])-len(atom_pos)
        for i in range(len(label_arr[b])):
            count[b, int(label_arr[b][i])] += 1
        for p in atom_pos:
            if count[b,p]>0: count[b,p]-=1
        for i in range(len(label_arr[b])):
            if count[b, int(label_arr[b][i])]>0: label[b,int(label_arr[b][i])] = 1
            else: label[b,int(label_arr[b][i])] = 0
    return label
def getGraph10(actions):
    graph = np.zeros((max_atoms+1, max_atoms+1))
    for action in actions:
        pos1,pos2,type = action[0],action[1],action[2]
        graph[pos1,pos2] += type
        graph[pos2, pos1] = graph[pos1,pos2]
    return graph
def accuracy(preds_bond,label_graph,vertex):
    bs = len(label_graph)
    preds_graph = np.zeros((bs,max_atoms,max_atoms)) #batch, max_atom=3, max_atom=3
    accs = []
    length = max_atoms #len(vertex[b])
    for b in range(bs):
        idx = 0
        acc = 0
        for i in range(length):
            for j in range(i+1, length):
                preds_graph[b,i,j] = preds_bond[b,idx]
                preds_graph[b, j, i] =preds_graph[b,i,j]
                idx += 1
                if preds_graph[b,i,j]==label_graph[b,i,j]:acc+=1
        accs.append(round(acc/(idx+np.finfo(np.float32).eps),4))
    return accs, preds_graph
def accuracy3(pred_atoms_type,labels_type,vertex):
    accs = []
    for b in range(len(pred_atoms_type)):
        acc = 0
        for i in range(len(vertex[b])):
            if pred_atoms_type[b][i]==labels_type[b][i]: acc+=1
        accs.append(round(acc/len(vertex[b]),4))
    return accs
def getLoss10(pred_graph, label_graph,vertex):
    acc = 0
    tot = 0
    for i in range(len(vertex)):
        num_H1 = pred_graph[i,max_atoms]
        num_H2 = bond_number[int(vertex[i])]-sum(label_graph[i])
        tot += num_H2
        acc+= min(num_H1,num_H2)
        for j in range(i+1,len(vertex)):
            if label_graph[i,j]!=0:
                tot+=label_graph[i,j]
                if label_graph[i,j]==pred_graph[i,j]: acc+= 1
    return 1-acc/tot
def accuracyGraph(pred,label):
    acc = 0
    tot = 0
    for i in range(max_atoms):
        for j in range(max_atoms):
            tot+=1
            if pred[i,j]==label[i,j]:acc+=1
    return acc/tot

#model
model = Classify9(padding_idx)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.NLLLoss(ignore_index=padding_idx) #CrossEntropyLoss()
#criterion = nn.MSELoss()

def train0(model,epoch,num):
    model.train()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num)- i)
        src = getInput0(vertex_data[i:i+seq_len],msp_arr_data[i:i+seq_len])
        labels = getLabel(mol_adj_data[i:i+seq_len],vertex_data[i:i+seq_len])
        labels_graph = mol_adj_data[i:i+seq_len]
        preds = model(src) #batch, 3, 4
        preds_bond = torch.argmax(preds, dim=2) #batch 3

        optimizer.zero_grad()
        loss = criterion(preds.view(-1, 4), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc, preds_graph = accuracy(preds_bond, labels_graph,vertex_data[i:i+seq_len])
        accs += acc
        if epoch % 100==0 and batch==0:
            print(labels_graph[0])
            print(preds_graph[0])
        # if epoch % 100 == 0:
        #     print(label_graph[0])
        #     print(preds_graph[0])
    print("epoch:", epoch)
    #print("accs: ", accs)
    print("train total_loss: ", round(total_loss,4))
    print("train mean_acc: ", round(sum(accs)/len(accs),4))
    return sum(accs)/len(accs)
def evaluate0(model,epoch,num):
    model.eval()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for i in range(0, len(num), batch_size):
            seq_len = min(batch_size, len(num) - i)
            src = getInput0(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            labels = getLabel(mol_adj_data[i:i + seq_len],vertex_data[i:i+seq_len])
            label_graph = mol_adj_data[i:i + seq_len]
            preds = model(src)  # batch, 3, 4
            preds_bond = torch.argmax(preds, dim=2)  # batch 3

            loss = criterion(preds.contiguous().view(-1, 4), labels.view(-1))
            total_loss += round(loss.item(), 4)
            acc, preds_graph = accuracy(preds_bond, label_graph,vertex_data[i:i+seq_len])
            accs += acc
            if epoch % 100 == 0:
                print(label_graph[0])
                print(preds_graph[0])
            # print(i)
        print("eval mean_accs: ", round(sum(accs)/len(accs),4))
        print("eval total_loss: ", round(total_loss,4))
def train1(model,epoch,num):
    model.train()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num)- i)
        src = getInput1(vertex_data[i:i+seq_len],msp_arr_data[i:i+seq_len])
        labels = getLabel(mol_adj_data[i:i+seq_len],vertex_data[i:i+seq_len])
        labels_graph = mol_adj_data[i:i+seq_len]
        preds = model(src) #batch, 3, 4
        preds_bond = torch.argmax(preds, dim=2) #batch 3

        optimizer.zero_grad()
        loss = criterion(preds.view(-1, 4), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc, preds_graph = accuracy(preds_bond, labels_graph,vertex_data[i:i+seq_len])
        accs += acc
        if epoch % 100==0 and batch==0:
            print(labels_graph[0])
            print(preds_graph[0])
        # if epoch % 100 == 0:
        #     print(label_graph[0])
        #     print(preds_graph[0])
    print("epoch:", epoch)
    #print("accs: ", accs)
    print("train total_loss: ", round(total_loss,4))
    print("train mean_acc: ", round(sum(accs)/len(accs),4))
    return sum(accs)/len(accs)
def evaluate1(model,epoch,num):
    model.eval()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for i in range(0, len(num), batch_size):
            seq_len = min(batch_size, len(num) - i)
            src = getInput1(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            labels = getLabel(mol_adj_data[i:i + seq_len],vertex_data[i:i+seq_len])
            label_graph = mol_adj_data[i:i + seq_len]
            preds = model(src)  # batch, 3, 4
            preds_bond = torch.argmax(preds, dim=2)  # batch 3

            loss = criterion(preds.contiguous().view(-1, 4), labels.view(-1))
            total_loss += round(loss.item(), 4)
            acc, preds_graph = accuracy(preds_bond, label_graph,vertex_data[i:i+seq_len])
            accs += acc
            if epoch % 100 == 0:
                print(label_graph[0])
                print(preds_graph[0])
            # print(i)
        print("eval mean_accs: ", round(sum(accs)/len(accs),4))
        print("eval total_loss: ", round(total_loss,4))
def train2(model,epoch,num):
    model.train()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num)- i)
        src = getInput2(vertex_data[i:i+seq_len],msp_arr_data[i:i+seq_len])
        labels = getLabel(mol_adj_data[i:i+seq_len],vertex_data[i:i+seq_len])
        labels_graph = mol_adj_data[i:i+seq_len]
        preds = model(src) #batch, 3, 4
        preds_bond = torch.argmax(preds, dim=2) #batch 3

        optimizer.zero_grad()
        loss = criterion(preds.view(-1, 4), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc, preds_graph = accuracy(preds_bond, labels_graph,vertex_data[i:i+seq_len])
        accs += acc
        if epoch % 100==0 and batch==0:
            print(labels_graph[0])
            print(preds_graph[0])
        # if epoch % 100 == 0:
        #     print(label_graph[0])
        #     print(preds_graph[0])
    print("epoch:", epoch)
    #print("accs: ", accs)
    print("train total_loss: ", round(total_loss,4))
    print("train mean_acc: ", round(sum(accs)/len(accs),4))
    return sum(accs)/len(accs)
def evaluate2(model,epoch,num):
    model.eval()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for i in range(0, len(num), batch_size):
            seq_len = min(batch_size, len(num) - i)
            src = getInput2(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            labels = getLabel(mol_adj_data[i:i + seq_len],vertex_data[i:i+seq_len])
            label_graph = mol_adj_data[i:i + seq_len]
            preds = model(src)  # batch, 3, 4
            preds_bond = torch.argmax(preds, dim=2)  # batch 3

            loss = criterion(preds.contiguous().view(-1, 4), labels.view(-1))
            total_loss += round(loss.item(), 4)
            acc, preds_graph = accuracy(preds_bond, label_graph,vertex_data[i:i+seq_len])
            accs += acc
            if epoch % 100 == 0:
                print(label_graph[0])
                print(preds_graph[0])
            # print(i)
        print("eval mean_accs: ", round(sum(accs)/len(accs),4))
        print("eval total_loss: ", round(total_loss,4))
def test2(model,num):
    print("Test!!!!!!!!!!!!!!!!!")
    model.eval()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for i in range(0, len(num), batch_size):
            seq_len = min(batch_size, len(num) - i)
            src = getInput2(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            labels = getLabel(mol_adj_data[i:i + seq_len])
            label_graph = mol_adj_data[i:i + seq_len]
            preds = model(src)  # batch, 3, 4
            preds_bond = torch.argmax(preds, dim=2)  # batch 3

            loss = criterion(preds.contiguous().view(-1, num_bonds_prediction), labels.view(-1))
            total_loss += round(loss.item(), 4)
            acc, preds_graph = accuracy(preds_bond, label_graph)
            accs += acc
            print(label_graph[0])
            print(preds_graph[0])
            #print(i)
        print("accs: ", accs)
        print("total_loss: ", total_loss)
#atom_type_predicted
def train3(model,epoch,num):
    model.train()
    vertex_data = vertex_arr[num]
    labels_data = labels_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num)- i)
        src = getInput3(vertex_data[i:i+seq_len],msp_arr_data[i:i+seq_len])
        labels = getLabel3(labels_data[i:i+seq_len])
        preds = model(src) #batch, 13, 28
        pred_atoms_type = torch.argmax(preds, dim=2) #batch 13

        optimizer.zero_grad()
        loss = criterion(preds.view(-1, 29), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc = accuracy3(pred_atoms_type, labels,vertex_data[i:i+seq_len])
        accs += acc
        if epoch % 10==0 and batch==0:
            print(labels[0])
            print(pred_atoms_type[0])
    print("epoch:", epoch)
    #print("accs: ", accs)
    print("train total_loss: ", round(total_loss,4))
    print("train mean_acc: ", round(sum(accs)/len(accs),4))
    return sum(accs)/len(accs)
def evaluate3(model,epoch,num):
    model.eval()
    vertex_data = vertex_arr[num]
    labels_data = labels_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for i in range(0, len(num), batch_size):
            seq_len = min(batch_size, len(num) - i)
            src = getInput3(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            labels = getLabel3(labels_data[i:i + seq_len])
            preds = model(src)  # batch, 13, 28
            pred_atoms_type = torch.argmax(preds, dim=2)  # batch 13
            loss = criterion(preds.view(-1, 29), labels.view(-1))
            total_loss += loss.item()
            acc = accuracy3(pred_atoms_type, labels, vertex_data[i:i + seq_len])
            accs += acc
            if epoch % 10 == 0 and i == 0:
                print(labels[0])
                print(pred_atoms_type[0])
        print("eval mean_accs: ", round(sum(accs)/len(accs),4))
        print("eval total_loss: ", round(total_loss,4))
def train4(model,epoch,num):
    model.train()
    vertex_data = vertex_arr[num]
    labels_data = labels_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num)- i)
        src = getInput4(vertex_data[i:i+seq_len],msp_arr_data[i:i+seq_len])
        labels = getLabel3(labels_data[i:i+seq_len])
        preds = model(src) #batch, 13, 28
        pred_atoms_type = torch.argmax(preds, dim=2) #batch 13

        optimizer.zero_grad()
        loss = criterion(preds.view(-1, 29), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc = accuracy3(pred_atoms_type, labels,vertex_data[i:i+seq_len])
        accs += acc
        if epoch % 10==0 and batch==0:
            print(labels[0])
            print(pred_atoms_type[0])
    print("epoch:", epoch)
    #print("accs: ", accs)
    print("train total_loss: ", round(total_loss,4))
    print("train mean_acc: ", round(sum(accs)/len(accs),4))
    return sum(accs)/len(accs)
def evaluate4(model,epoch,num):
    model.eval()
    vertex_data = vertex_arr[num]
    labels_data = labels_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for i in range(0, len(num), batch_size):
            seq_len = min(batch_size, len(num) - i)
            src = getInput4(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            labels = getLabel3(labels_data[i:i + seq_len])
            preds = model(src)  # batch, 13, 28
            pred_atoms_type = torch.argmax(preds, dim=2)  # batch 13
            loss = criterion(preds.view(-1, 29), labels.view(-1))
            total_loss += loss.item()
            acc = accuracy3(pred_atoms_type, labels, vertex_data[i:i + seq_len])
            accs += acc
            if epoch % 10 == 0 and i == 0:
                print(labels[0])
                print(pred_atoms_type[0])
        print("eval mean_accs: ", round(sum(accs)/len(accs),4))
        print("eval total_loss: ", round(total_loss,4))
#imitation
def train5(model,epoch,num):
    model.train()
    vertex_data = vertex_arr[num]
    labels_data = labels_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num)- i)
        actions = {}
        pos_prob = []
        labels_ori = getLabel5(labels_data[i:i + seq_len],[]).numpy()
        while(len(actions)<len(vertex_data[i])):
            src = getInput5(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len],actions)
            #labels = getLabel5(labels_data[i:i + seq_len],actions.keys())
            labels = getLabel5(labels_data[i:i + seq_len], [])
            preds,masks = model(src)  # batch, 13, 29
            preds_prob = F.log_softmax(preds,dim=2)
            #choose one action
            all_probs = F.softmax(preds.masked_fill(masks, -1e9).view(seq_len,-1),dim=1)
            action = torch.argmax(all_probs[0])
            # m = torch.distributions.Categorical(all_probs[0])
            # action = m.sample()
            pos = int(action//29)
            type = int(action%29)
            if pos not in actions.keys():
                actions[pos] = type
        optimizer.zero_grad()
        loss = criterion(preds_prob.view(-1, 29), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        #print(actions)
        pred_atoms_type = np.zeros_like(labels_ori)
        for j in range(len(vertex_data[i])):
            pred_atoms_type[0,j] = actions[j]
        acc = accuracy3(pred_atoms_type, labels_ori,vertex_data[i:i+seq_len])
        accs += acc
        if epoch % 1==0 and batch==0:
            print(labels_ori[0])
            print(pred_atoms_type[0])
    print("epoch:", epoch)
    #print("accs: ", accs)
    print("train total_loss: ", round(total_loss,4))
    print("train mean_acc: ", round(sum(accs)/len(accs),4))
    return sum(accs)/len(accs)
def evaluate5(model,epoch,num):
    model.eval()
    vertex_data = vertex_arr[num]
    labels_data = labels_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for i in range(0, len(num), batch_size):
            seq_len = min(batch_size, len(num) - i)
            actions = {}
            pos_prob = []
            labels_ori = getLabel5(labels_data[i:i + seq_len], []).numpy()
            while (len(actions) < len(vertex_data[i])):
                src = getInput5(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len], actions)
                # labels = getLabel5(labels_data[i:i + seq_len],actions.keys())
                labels = getLabel5(labels_data[i:i + seq_len], [])
                preds, masks = model(src)  # batch, 13, 29
                preds_prob = F.log_softmax(preds, dim=2)

                # choose one action
                all_probs = F.softmax(preds.masked_fill(masks, -1e9).view(seq_len, -1), dim=1)
                action = torch.argmax(all_probs[0])
                # m = torch.distributions.Categorical(all_probs[0])
                # action = m.sample()
                pos = int(action // 29)
                type = int(action % 29)
                if pos not in actions.keys():
                    actions[pos] = type
            loss = criterion(preds_prob.view(-1, 29), labels.view(-1))
            total_loss += loss.item()
            pred_atoms_type = np.zeros_like(labels_ori)
            for j in range(len(vertex_data[i])):
                pred_atoms_type[0, j] = actions[j]
            acc = accuracy3(pred_atoms_type, labels_ori, vertex_data[i:i + seq_len])
            accs += acc
            if epoch % 1 == 0 and i == 0:
                print(labels_ori[0])
                print(pred_atoms_type[0])
        print("eval mean_accs: ", round(sum(accs)/len(accs),4))
        print("eval total_loss: ", round(total_loss,4))
def train6(model,epoch,num):
    model.train()
    vertex_data = vertex_arr[num]
    labels_data = labels_arr[num]
    msp_arr_data = msp_arr[num]
    h_num_data = H_num[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num)- i)
        actions = []
        atom_pos = []
        labels_ori = getLabel5(labels_data[i:i + seq_len],[]).numpy()
        idx = 0#len(vertex_data[i])
        log_probs = []
        #print(labels_ori)
        while(idx<len(vertex_data[i])):
            #[vertex, msp, h_num, actions):  # imitation #actions = {atom_pos:reward,...} h_num=[10,2,...]
            src = getInput6(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len],h_num_data[i:i+seq_len],atom_pos)
            labels = getLabel6(labels_data[i:i + seq_len], atom_pos)
            preds = model(src)  # batch, 28 predicted rewards
            #optimizer.zero_grad()
            #loss = criterion(torch.sigmoid(preds).view(-1), labels.view(-1))
            #print(loss.item())
            #loss.backward()
            # optimizer.step()
            # total_loss += loss.item()

            probs = F.softmax(preds,dim=1)
            m = torch.distributions.Categorical(probs[0])
            pos = m.sample()
            #pos = torch.argmax(preds,dim=1)
            # min_mse = 1e9
            # pos,reward = 0,0,
            # for ii in range(num_bonds_prediction):
            #     if math.sqrt((preds[0,ii]-idx)**2)<min_mse:
            #         min_mse = math.sqrt((preds[0,ii]-idx)**2)
            #         pos,reward = ii,float(preds[0,ii])
            #actions[pos] = pos
            #print(atom_pos)
            atom_pos.append(int(pos))
            actions.append([int(pos),torch.log(probs[0][pos])])
            idx+=1
        actions.sort(key=lambda r: r[0])
        atom_pos.sort()
        losses = []
        penalty = 0
        for ii in range(len(actions)):
            losses.append(-1 * actions[ii][1])
            losses.append(-1 * actions[ii][1])
            if actions[ii][0]!=labels_ori[0][ii]:
                penalty += 1
        loss = torch.stack(losses).sum()*penalty
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # pred_atoms_type = np.zeros_like(labels_ori)
        # for j in range(len(vertex_data[i])):
        #     pred_atoms_type[0,j] = actions[j]
        # acc = accuracy3(pred_atoms_type, labels_ori,vertex_data[i:i+seq_len])
        # accs += acc
        if epoch % 1==0 and batch==0:
            print(labels_ori[0])
            print(atom_pos)
            print(loss)
            #print(pred_atoms_type[0])
    print("epoch:", epoch)
    #print("accs: ", accs)
    print("train total_loss: ", round(total_loss,4))
    #print("train mean_acc: ", round(sum(accs)/len(accs),4))
    #return sum(accs)/len(accs)
def evaluate6(model,epoch,num):
    model.eval()
    vertex_data = vertex_arr[num]
    labels_data = labels_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for i in range(0, len(num), batch_size):
            seq_len = min(batch_size, len(num) - i)
            actions = {}
            idx = 0
            labels_ori = getLabel5(labels_data[i:i + seq_len], []).numpy()
            while (len(actions) < len(vertex_data[i])):
                idx += 1
                src = getInput5(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len], actions)
                #labels = getLabel5(labels_data[i:i + seq_len], actions.keys())
                labels = getLabel5(labels_data[i:i + seq_len],[])
                preds, masks = model(src)  # batch, 13, 29
                preds_prob = F.log_softmax(preds, dim=2)
                # loss = criterion(preds_prob.view(-1, 29), labels.view(-1))
                # total_loss += loss.item()
                # choose one action
                all_probs = F.softmax(preds.masked_fill(masks, -1e9).view(seq_len, -1), dim=1)
                action = torch.argmax(all_probs[0])
                pos = int(action // 29)
                type = int(action % 29)
                if pos not in actions.keys(): actions[pos] = type
            loss = criterion(preds_prob.view(-1, 29), labels.view(-1))
            total_loss += loss.item()
            pred_atoms_type = np.zeros_like(labels_ori)
            for j in range(len(vertex_data[i])):
                pred_atoms_type[0, j] = actions[j]
            acc = accuracy3(pred_atoms_type,labels_ori, vertex_data[i:i + seq_len])
            accs += acc
            if epoch % 1 == 0 and i==0:
                print(labels_ori[0])
                print(pred_atoms_type[0])
        print("eval mean_accs: ", round(sum(accs)/len(accs),4))
        print("eval total_loss: ", round(total_loss,4))
def train7(model,epoch,num):
    model.train()
    vertex_data = vertex_arr[num]
    labels_data = labels_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num)- i)
        actions = {}
        pos_prob = []
        labels = getLabel(mol_adj_data[i:i+seq_len],vertex_data[i:i+seq_len])
        labels_graph = mol_adj_data[i:i + seq_len]
        while(len(actions)<edge_num):
            src = getInput7(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len],actions)
            preds,mask = model(src)  # batch, edge_num, 4
            preds_prob = F.log_softmax(preds,dim=2)
            #choose one action
            all_probs = F.softmax(preds.masked_fill(mask, -1e9).view(seq_len,-1),dim=1)
            #action = torch.argmax(all_probs[0])
            m = torch.distributions.Categorical(all_probs[0])
            action = m.sample()
            pos = int(action//4)
            type = int(action%4)
            if pos not in actions.keys():
                actions[pos] = type
        optimizer.zero_grad()
        loss = criterion(preds_prob.view(-1, 4), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        #print(actions)
        preds_bond = []
        for p in range(edge_num):
            preds_bond.append(actions[p])
        acc, preds_graph = accuracy(torch.Tensor([preds_bond]), labels_graph, vertex_data[i:i + seq_len])
        accs += acc
        if epoch % 1 == 0 and batch%1 == 0:
            print(labels_graph[0])
            print(preds_graph[0])
    print("epoch:", epoch)
    #print("accs: ", accs)
    print("train total_loss: ", round(total_loss,4))
    print("train mean_acc: ", round(sum(accs)/len(accs),4))
    return sum(accs)/len(accs)
def evaluate7(model,epoch,num):
    model.eval()
    vertex_data = vertex_arr[num]
    labels_data = labels_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for i in range(0, len(num), batch_size):
            seq_len = min(batch_size, len(num) - i)
            actions = {}
            pos_prob = []
            labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
            labels_graph = mol_adj_data[i:i + seq_len]
            while (len(actions) < edge_num):
                src = getInput7(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len], actions)
                preds, mask = model(src)  # batch, edge_num, 4
                preds_prob = F.log_softmax(preds, dim=2)
                # choose one action
                all_probs = F.softmax(preds.masked_fill(mask, -1e9).view(seq_len, -1), dim=1)
                #action = torch.argmax(all_probs[0])
                m = torch.distributions.Categorical(all_probs[0])
                action = m.sample()
                pos = int(action // 4)
                type = int(action % 4)
                if pos not in actions.keys():
                    actions[pos] = type
            loss = criterion(preds_prob.view(-1, 4), labels.view(-1))
            total_loss += loss.item()
            # print(actions)
            preds_bond = []
            for p in range(edge_num):
                preds_bond.append(actions[p])
            acc, preds_graph = accuracy(torch.Tensor([preds_bond]), labels_graph, vertex_data[i:i + seq_len])
            accs += acc
            if epoch % 1 == 0 and i%1 == 0:
                print(labels_graph[0])
                print(preds_graph[0])
        print("eval mean_accs: ", round(sum(accs)/len(accs),4))
        print("eval total_loss: ", round(total_loss,4))
#transformer with linear
def train8(model,epoch,num):
    model.train()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num)- i)
        src = getInput0(vertex_data[i:i+seq_len],msp_arr_data[i:i+seq_len])
        labels = getLabel(mol_adj_data[i:i+seq_len],vertex_data[i:i+seq_len])
        labels_graph = mol_adj_data[i:i+seq_len]
        preds = model(src) #batch, 3, 4
        preds_bond = torch.argmax(preds, dim=2) #batch 3

        optimizer.zero_grad()
        loss = criterion(preds.view(-1, 4), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc, preds_graph = accuracy(preds_bond, labels_graph,vertex_data[i:i+seq_len])
        accs += acc
        if epoch % 100==0 and batch==0:
            print(labels_graph[0])
            print(preds_graph[0])
        # if epoch % 100 == 0:
        #     print(label_graph[0])
        #     print(preds_graph[0])
    print("epoch:", epoch)
    #print("accs: ", accs)
    print("train total_loss: ", round(total_loss,4))
    print("train mean_acc: ", round(sum(accs)/len(accs),4))
    return sum(accs)/len(accs)
def evaluate8(model,epoch,num):
    model.eval()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for i in range(0, len(num), batch_size):
            seq_len = min(batch_size, len(num) - i)
            src = getInput0(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            labels = getLabel(mol_adj_data[i:i + seq_len],vertex_data[i:i+seq_len])
            label_graph = mol_adj_data[i:i + seq_len]
            preds = model(src)  # batch, 3, 4
            preds_bond = torch.argmax(preds, dim=2)  # batch 3

            loss = criterion(preds.contiguous().view(-1, 4), labels.view(-1))
            total_loss += round(loss.item(), 4)
            acc, preds_graph = accuracy(preds_bond, label_graph,vertex_data[i:i+seq_len])
            accs += acc
            if epoch % 100 == 0:
                print(label_graph[0])
                print(preds_graph[0])
            # print(i)
        print("eval mean_accs: ", round(sum(accs)/len(accs),4))
        print("eval total_loss: ", round(total_loss,4))
#transformer with edge
def train9(model,epoch,num):
    model.train()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num)- i)
        src = getInput9(vertex_data[i:i+seq_len],msp_arr_data[i:i+seq_len])
        labels = getLabel(mol_adj_data[i:i+seq_len],vertex_data[i:i+seq_len])
        labels_graph = mol_adj_data[i:i+seq_len]
        preds = model(src) #batch, edge_num, 4
        preds_bond = torch.argmax(preds, dim=-1) #batch edge_num

        optimizer.zero_grad()
        loss = criterion(preds.view(-1, 4), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc, preds_graph = accuracy(preds_bond, labels_graph,vertex_data[i:i+seq_len])
        accs += acc
        if epoch % 100==0 and batch==0:
            print(labels_graph[0])
            print(preds_graph[0])
    print("epoch:", epoch)
    #print("accs: ", accs)
    print("train total_loss: ", round(total_loss,4))
    print("train mean_acc: ", round(sum(accs)/len(accs),4))
    return sum(accs)/len(accs)
def evaluate9(model,epoch,num):
    model.eval()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for i in range(0, len(num), batch_size):
            seq_len = min(batch_size, len(num) - i)
            src = getInput9(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            labels = getLabel(mol_adj_data[i:i + seq_len],vertex_data[i:i+seq_len])
            label_graph = mol_adj_data[i:i + seq_len]
            preds = model(src)  # batch, 3, 4
            preds_bond = torch.argmax(preds, dim=-1)  # batch 3

            loss = criterion(preds.contiguous().view(-1, 4), labels.view(-1))
            total_loss += round(loss.item(), 4)
            acc, preds_graph = accuracy(preds_bond, label_graph,vertex_data[i:i+seq_len])
            accs += acc
            if epoch % 100 == 0:
                print(label_graph[0])
                print(preds_graph[0])
            # print(i)
        print("eval mean_accs: ", round(sum(accs)/len(accs),4))
        print("eval total_loss: ", round(total_loss,4))
#imitation linear
def train10(model,epoch,num):
    model.train()
    vertex_data = vertex_arr[num]
    label_graph = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    h_num_data = H_num[num]
    total_loss = 0
    accs = []
    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num)- i)
        actions = [[0,1,2],[0,2,1]] #[ [pos1, pos2, type] ]
        log_probs = []
        while(1):
            #print(actions)
            src,flag = getInput10(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len],h_num_data[i:i+seq_len],actions)
            #print(src)
            if flag==0: break
            probs = model(src)  # batch, (edge_num+atom_max) * 3
            probs = F.softmax(probs.view(src.size(0), -1), dim=-1)
            m = torch.distributions.Categorical(probs[0])
            pos = m.sample()
            pos1, pos2 = getInput.find(pos // 3,max_atoms+1)
            type = pos % 3+1
            actions.append([pos1,pos2,int(type)])
            log_probs.append(torch.log(probs[0][pos]))
        pred_graph = getGraph10(actions)
        #atom_lists = getInput.find_permutation(vertex_data[i], start=3)
        min_loss = getLoss10(pred_graph, label_graph[i], vertex_data[i])
        # best_graph = pred_graph
        # print(label_graph[i])
        # for indexs in range(len(atom_lists)):
        #     new_E = np.zeros((max_atoms+1,max_atoms+1))
        #     for ii in range(len(atom_lists[indexs])):  # per atom
        #         for jj in range(len(atom_lists[indexs])):  # per atom
        #             new_E[ii, jj] = pred_graph[atom_lists[indexs][ii], atom_lists[indexs][jj]]
        #         new_E[ii,max_atoms] = bond_number[int(vertex_data[i][ii])]-sum(new_E[ii])
        #     temp = getLoss10(new_E, label_graph[i], vertex_data[i])
        #     if temp<min_loss:
        #         min_loss = temp
        #         best_graph = new_E
        # print(best_graph)
        # pred_graph = best_graph
        loss = torch.stack(log_probs).sum()*(-1*min_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        acc = accuracyGraph(pred_graph,label_graph[i])
        accs.append(acc)
        if epoch % 1==0 and i==0:
            print(label_graph[i])
            print(pred_graph)
            #print(pred_atoms_type[0])
    print("epoch:", epoch)
    print("train total_loss: ", round(total_loss,4))
    print("Average accuracy:", round(sum(accs)/len(accs),4))
def evaluate10(model,epoch,num):
    model.eval()
    vertex_data = vertex_arr[num]
    label_graph = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    h_num_data = H_num[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for batch, i in enumerate(range(0, len(num), batch_size)):
            seq_len = min(batch_size, len(num) - i)
            actions = [[0,1,2],[0,2,1]] #[ [pos1, pos2, type] ]
            log_probs = []
            while (1):
                # print(actions)
                src, flag = getInput10(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len],
                                       h_num_data[i:i + seq_len], actions)
                # print(src)
                if flag == 0: break
                probs = model(src)  # batch, (edge_num+atom_max) * 3
                probs = F.softmax(probs.view(src.size(0), -1), dim=-1)
                m = torch.distributions.Categorical(probs[0])
                pos = m.sample()
                pos1, pos2 = getInput.find(pos // 3, max_atoms + 1)
                type = pos % 3 + 1
                actions.append([pos1, pos2, int(type)])
                log_probs.append(torch.log(probs[0][pos]))
            pred_graph = getGraph10(actions)
            loss = torch.stack(log_probs).sum() * (-getLoss10(pred_graph, label_graph[i], vertex_data[i]))
            total_loss += loss.item()
            acc = accuracyGraph(pred_graph, label_graph[i])
            accs.append(acc)
            if epoch % 1 == 1000 and i == 0:
                print(label_graph[i])
                print(pred_graph)
        print("epoch:", epoch)
        print("Eval total_loss: ", round(total_loss, 4))
        print("Average accuracy:", round(sum(accs) / len(accs), 4))
def train11(model,epoch,num): #transformer with reinforcement learning
    model.train()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num) - i)
        src = getInput9(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
        labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
        labels_graph = mol_adj_data[i:i + seq_len]
        probs = model(src)  # batch, edge_num, 4
        m = torch.distributions.Categorical(probs)
        preds_bond = m.sample() #edge_num
        #preds_bond = torch.argmax(probs, dim=-1)  # batch edge_num
        min_loss = 0.0
        tot = 0
        log_probs = []
        for ii in range(len(preds_bond[0])):
            if labels[0][ii]!=padding_idx:
                tot+=1
                #log_probs.append(torch.log(probs[0][ii][preds_bond[0][ii]]))
                if int(preds_bond[0][ii]) != int(labels[0][ii]):
                    #min_loss+=1.0
                    log_probs.append(1)
        loss = torch.Tensor(log_probs).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc, preds_graph = accuracy(preds_bond, labels_graph, vertex_data[i:i + seq_len])
        accs += acc
        if epoch % 100 == 0 and batch== 0:
            print(labels_graph[0])
            print(preds_graph[0])
    print("epoch:", epoch)
    print("train total_loss: ", round(total_loss, 4))
    print("train mean_acc: ", round(sum(accs) / len(accs), 4))
    return sum(accs) / len(accs)
def evaluate11(model,epoch,num):
    model.eval()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for batch, i in enumerate(range(0, len(num), batch_size)):
            seq_len = min(batch_size, len(num) - i)
            src = getInput9(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
            labels_graph = mol_adj_data[i:i + seq_len]
            preds = model(src)  # batch, edge_num, 4
            preds_bond = torch.argmax(preds, dim=-1)  # batch edge_num
            probs, _ = torch.max(preds, dim=-1)  # batch edge_num
            min_loss = 0
            tot = 0
            log_probs = []
            for ii in range(len(preds_bond[0])):
                if labels[0][ii] != padding_idx:
                    tot += 1
                    log_probs.append(torch.log(probs[0][ii]))
                    if int(preds_bond[0][ii]) != int(labels[0][ii]): min_loss += 1
            loss = torch.stack(log_probs).sum() * (-min_loss)
            total_loss += loss.item()
            acc, preds_graph = accuracy(preds_bond, labels_graph, vertex_data[i:i + seq_len])
            accs += acc
            if epoch % 100 == 0 and batch == 0:
                print(labels_graph[0])
                print(preds_graph[0])
        print("eval total_loss: ", round(total_loss, 4))
        print("eval mean_acc: ", round(sum(accs) / len(accs), 4))
        return sum(accs) / len(accs)

def train_linear(epoch,num):
    for i in range(1,1+epoch):
        train11(model,i,num)
        #if acc>=1: break
        #evaluate11(model, i,range(1700,1708))
    #torch.save(model.state_dict(),'model_linear2.pkl')
    #model.load_state_dict(torch.load('model5.pkl'))
    #test_model(model,range(16,24))


train_linear(1000,num=range(8))

#Testing
# model.load_state_dict(torch.load('model_linear2.pkl'))
# test_model(model,range(160, 168))