import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import copy
from tsfm.Layers import EncoderLayer
from tsfm.Sublayers import Norm
from tsfm.Embed import PositionalEncoder
# import matplotlib.pyplot as plt

vertex_arr = np.load("../tsfm/vertex_arr_sort_per.npy", allow_pickle=True) #1843
# 1843 molecules, each stores [0, 2, 2, 0]

mol_adj_arr = np.load("../tsfm/mol_adj_arr_sort_per.npy", allow_pickle=True)
# (1843, 13, 13) adjacency matrix of the bonds

msp_arr = np.load("../tsfm/msp_arr_sort_per.npy", allow_pickle=True)
# (1843, 800)
H_num = np.load("../tsfm/h_num_per.npy", allow_pickle=True)
# (1843, 2)

msp_len = 800
k = 20
atom_type = 2
padding_idx = 799  # must > 0 and < msp_len
dropout = 0.1
batch_size = 8
atom_mass = [12, 1, 16, 14]  # [12,1,16]
atomic_number = [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]]  # [C, H, O, N}
bond_number = [4, 1, 2]  # [C, H, O]
default_valence = [[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
atom_pos_mass = [28, 29, 33, 40, 32 + 12, 16 * 3, 12 + 4, 24 + 1, 24 + 2, 24 + 3, 28 + 2, 28 + 3, 12 * 3, 36 + 1, 38,
                 24 + 16, 41, 41, 42, 48, 49, 36 + 16, 52, 53, 28 * 2, 56, 57, 12 * 5]

edge_num = 78  # 3#29*15#78 #3
d_model = 256
max_atoms = 13  # 3
max_len11 = 3 * edge_num + k
max_len12 = 2 * edge_num + k

def  countH(h_num):
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
def getEdgeIdx(pos1, pos2=None):  # not contain H
    edge_idx = 0
    for jj in range(pos1):
        edge_idx += (max_atoms - jj - 1)
    if pos2 == None:
        return edge_idx - 1
    else:
        edge_idx += pos2 - pos1
    return edge_idx - 1
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx)  # .unsqueeze(-2)
def get_pad_mask10(seq,vertex):
    mask = torch.ByteTensor([[[False] * 4] * edge_num] * seq.size(0))  # torch.zeros((seq.size(0),seq.size(1),4))
    for b in range(seq.size(0)):
        length = len(vertex[b])
        idx = 0
        for i in range(max_atoms):
            for j in range(i + 1, max_atoms):
                if i >= length or j >= length: mask[b, idx] = torch.ByteTensor([0, 1, 1, 1])
                idx += 1
    return mask
def get_pad_mask11(seq):
    mask =  torch.ByteTensor([[[False]*4]*edge_num]*seq.size(0)) #torch.zeros((seq.size(0),seq.size(1),4))
    for b in range(seq.size(0)):
        for i in range(edge_num):
            if seq[b,i*3]==padding_idx: mask[b,i] = torch.ByteTensor([0,1,1,1])
            if seq[b, i * 3] == 16 or seq[b, i * 3+1] == 16 : mask[b, i] = torch.ByteTensor([0, 0, 0, 1])
    return mask
def get_pad_mask12(seq):
    mask =  torch.ByteTensor([[[False]*4]*edge_num]*seq.size(0)) #torch.zeros((seq.size(0),seq.size(1),4))
    for b in range(seq.size(0)):
        for i in range(edge_num):
            if seq[b,i*2]==padding_idx: mask[b,i] = torch.ByteTensor([0,1,1,1])
            if seq[b, i * 2] == 16 or seq[b, i * 2+1] == 16 : mask[b, i] = torch.ByteTensor([0, 0, 0, 1])
    return mask

class Classify10(nn.Module):  # transformer with linear using ms only
    def __init__(self, padding_idx):
        super().__init__()
        heads = 4
        self.N = 3
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len, d_model, self.padding_idx)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), self.N)
        self.norm = Norm(d_model)
        self.ll1 = nn.Linear(d_model, edge_num)
        self.ll2 = nn.Linear(k, 4)

    def forward(self, src,vertex):  # [batch, atom_num=13, per_lin]
        self.mask = get_pad_mask10(src, vertex)  # [batch, edge_num,4]
        mask = get_pad_mask(src, self.padding_idx).view(src.size(0), -1).unsqueeze(-1)
        output = self.embedding(src)  # [batch, k, d_model=512]
        for i in range(self.N):
            output = self.layers[i](output, mask)
        output = self.ll1(output)  # [batch, k, edge_num]
        output = output.permute(0, 2, 1)  # [batch, edge_num, k]
        output = self.ll2(output)  # [batch, edge_num, 4]
        output = output.masked_fill(self.mask, -1e9)
        output = F.log_softmax(output, dim=2)
        return output  # [batch, edge_num=3, bond=4]
class Classify11(nn.Module):  # transformer with edge as well
    def __init__(self, padding_idx):
        super().__init__()
        heads = 4
        self.N = 1
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len, d_model, self.padding_idx)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), self.N)
        self.norm = Norm(d_model)
        self.ll1 = nn.Linear(d_model, edge_num)
        self.ll2 = nn.Linear(max_len11, 4)

    def forward(self, src):
        self.mask = get_pad_mask11(src)  # [batch, edge_num,4]
        mask = get_pad_mask(src, self.padding_idx).view(src.size(0), -1).unsqueeze(-1)
        output = self.embedding(src)  # [batch, k, d_model=512]
        for i in range(self.N):
            output = self.layers[i](output, mask)
        output = self.ll1(output)  # [batch, max_len2, edge_num]
        output = output.permute(0, 2, 1)  # [batch, edge_num, max_len2]
        output = self.ll2(output)  # [batch, edge_num, 4]
        output = output.masked_fill(self.mask,-1e9)
        output = F.log_softmax(output, dim=-1)
        return output  # [batch, edge_num=3, bond=4]
class Classify12(nn.Module):  # transformer with edge as well
    def __init__(self, padding_idx):
        super().__init__()
        heads = 4
        self.N = 1
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len, d_model, self.padding_idx)
        self.pe = PositionalEncoder(d_model=d_model, max_seq_len=edge_num * 2, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), self.N)
        self.norm = Norm(d_model)
        self.ll1 = nn.Linear(d_model, edge_num)
        self.ll2 = nn.Linear(max_len12, 4)

    def forward(self, src):
        self.mask = get_pad_mask12(src)  # [batch, edge_num,4]
        mask = get_pad_mask(src, self.padding_idx).view(src.size(0), -1).unsqueeze(-1)
        output = self.embedding(src)  # [batch, k, d_model=512]
        output = torch.cat((self.pe(output[:, :2 * edge_num]), output[:, 2 * edge_num:]), dim=1)
        for i in range(self.N):
            output = self.layers[i](output, mask)
        output = self.ll1(output)  # [batch, max_len12, edge_num]
        output = output.permute(0, 2, 1)  # [batch, edge_num, max_len12]
        output = self.ll2(output)  # [batch, edge_num, 4]
        output = output.masked_fill(self.mask,-1e9)
        output = F.log_softmax(output, dim=-1)
        return output  # [batch, edge_num=3, bond=4]
class Classify31(nn.Module):  # transformer with edge as well
    def __init__(self, padding_idx):
        super().__init__()
        heads = 4
        self.N = 1
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len, d_model, self.padding_idx)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), self.N)
        self.norm = Norm(d_model)
        self.ll1 = nn.Linear(d_model, edge_num)
        self.ll2 = nn.Linear(max_len11, 4)

    def forward(self, src):
        self.mask = get_pad_mask11(src)  # [batch, edge_num,4]
        mask = get_pad_mask(src, self.padding_idx).view(src.size(0), -1).unsqueeze(-1)
        output = self.embedding(src)  # [batch, k, d_model=512]
        for i in range(self.N):
            output = self.layers[i](output, mask)
        output = self.ll1(output)  # [batch, max_len2, edge_num]
        output = output.permute(0, 2, 1)  # [batch, edge_num, max_len2]
        output = self.ll2(output)  # [batch, edge_num, 4]
        output = output.masked_fill(self.mask,-1e9)
        output = F.softmax(output, dim=-1)
        return output  # [batch, edge_num=3, bond=4]
class Classify32(nn.Module):  # transformer with edge as well
    def __init__(self, padding_idx):
        super().__init__()
        heads = 4
        self.N = 1
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len, d_model, self.padding_idx)
        self.pe = PositionalEncoder(d_model=d_model, max_seq_len=edge_num * 2, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), self.N)
        self.norm = Norm(d_model)
        self.ll1 = nn.Linear(d_model, edge_num)
        self.ll2 = nn.Linear(max_len12, 4)

    def forward(self, src):
        self.mask = get_pad_mask12(src)  # [batch, edge_num,4]
        mask = get_pad_mask(src, self.padding_idx).view(src.size(0), -1).unsqueeze(-1)
        output = self.embedding(src)  # [batch, k, d_model=512]
        output = torch.cat((self.pe(output[:, :2 * edge_num]), output[:, 2 * edge_num:]), dim=1)
        for i in range(self.N):
            output = self.layers[i](output, mask)
        output = self.ll1(output)  # [batch, max_len12, edge_num]
        output = output.permute(0, 2, 1)  # [batch, edge_num, max_len12]
        output = self.ll2(output)  # [batch, edge_num, 4]
        output = output.masked_fill(self.mask,-1e9)
        output = F.softmax(output, dim=-1)
        return output  # [batch, edge_num=3, bond=4]

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
def getInput11(vertex, msp): #transformer without positional encoding
    # [A1-weight, A2-weight, pos,A1, A3,pos  A2, A3,pos, msp1[x], ...,mspk[x]] 45
    src = torch.LongTensor([[padding_idx] * max_len11] * len(vertex))  # [batch, 45]
    for b in range(len(vertex)):
        idx1 = 0
        for i in range(max_atoms):
            for ii in range(i + 1, max_atoms):
                if i < len(vertex[b]) and ii < len(vertex[b]):
                    src[b, idx1 * 3], src[b, idx1 * 3 + 1] = atom_mass[int(vertex[b][i])], atom_mass[int(vertex[b][ii])]
                    src[b, idx1 * 3 + 2] = idx1 + 1
                idx1 += 1
        idx = 3 * edge_num
        for j in range(1, len(msp[b]) - 1):
            if msp[b][j - 1] < msp[b][j] and msp[b][j + 1] < msp[b][j]:
                src[b, idx] = j
                idx += 1
                if idx == max_len11: break
    return src
def getInput12(vertex, msp): #transformer with positional encoding
    # [A1-weight, A2-weight, A1, A3, A2, A3, msp1[x], ...,mspk[x]] 45
    src = torch.LongTensor([[padding_idx] * max_len12] * len(vertex))  # [batch, 45]
    for b in range(len(vertex)):
        idx1 = 0
        for i in range(max_atoms):
            for ii in range(i + 1, max_atoms):
                if i < len(vertex[b]) and ii < len(vertex[b]):
                    src[b, idx1 * 2], src[b, idx1 * 2 + 1] = atom_mass[int(vertex[b][i])], atom_mass[int(vertex[b][ii])]
                idx1 += 1
        idx = 2 * edge_num
        for j in range(1, len(msp[b]) - 1):
            if msp[b][j - 1] < msp[b][j] and msp[b][j + 1] < msp[b][j]:
                src[b, idx] = j
                idx += 1
                if idx == max_len12: break
    return src

def getLabel(mol_arr, vertex):
    # label = torch.zeros((len(mol_arr),edge_num),dtype=torch.long) #[batch, edge_num]
    label = torch.LongTensor([[padding_idx] * edge_num] * len(mol_arr))
    for b in range(len(mol_arr)):  # batch
        idx = 0
        for i in range(len(mol_arr[b])):
            for j in range(i + 1, len(mol_arr[b])):
                if i < len(vertex[b]) and j < len(vertex[b]):
                    label[b, idx] = mol_arr[b][i][j]
                idx += 1
    return label
def accuracy(preds_bond,label_graph,vertex):
    bs = len(label_graph)
    preds_graph = torch.zeros((bs,max_atoms,max_atoms)) #batch, max_atom=3, max_atom=3
    accs = []
    for b in range(bs):
        idx = 0
        acc = 0
        count = 0
        length = len(vertex[b])
        for i in range(max_atoms):
            for j in range(i+1, max_atoms):
                preds_graph[b,i,j] = preds_bond[b,idx]
                preds_graph[b, j, i] = preds_graph[b,i,j]
                idx +=1
                if i < length and j < length and [i,j] not in [[0,1],[0,2]]:
                    count += 1
                    if preds_graph[b,i,j]==label_graph[b,i,j]:acc+=1
        accs.append(round(acc/(count+np.finfo(np.float32).eps),4))
    return accs, preds_graph
# model
model = Classify31(padding_idx)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.NLLLoss(ignore_index=padding_idx)  # CrossEntropyLoss()

def reinforceLoss(probs,labels):
    m = torch.distributions.Categorical(probs)
    preds_bond = m.sample()  # edge_num
    loss = 0
    for b in range(len(preds_bond)):
        log_prob = []
        reward = []
        for ii in range(len(preds_bond[b])):
            if labels[0][ii] != padding_idx:
                bond_type = preds_bond[b][ii]
                log_prob.append(torch.log(probs[b][ii][bond_type]))
                if int(preds_bond[b][ii]) == int(labels[b][ii]):
                    reward.append(10)
                else:
                    reward.append(1)
        ave_reward = sum(reward) / len(reward)
        loss += torch.Tensor(torch.stack(log_prob)).sum() * (-ave_reward)
    return loss/len(preds_bond),preds_bond
# transformer with linear
def train10(model, epoch, num):
    vertex_arr = np.load("../tsfm/vertex_arr_sort_svd.npy", allow_pickle=True)  # 1843
    mol_adj_arr = np.load("../tsfm/mol_adj_arr_sort_svd.npy", allow_pickle=True)
    msp_arr = np.load("../tsfm/msp_arr_sort.npy", allow_pickle=True)
    model.train()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num) - i)
        src = getInput0(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
        labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
        labels_graph = mol_adj_data[i:i + seq_len]
        preds = model(src,vertex_data[i:i + seq_len])  # batch, 3, 4
        preds_bond = torch.argmax(preds, dim=2)  # batch 3

        optimizer.zero_grad()
        loss = criterion(preds.view(-1, 4), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc, preds_graph = accuracy(preds_bond, labels_graph, vertex_data[i:i + seq_len])
        accs += acc
        if (epoch - 1) % 50 == 0 and batch==0:
            print(labels_graph[0])
            print(preds_graph[0])
    print("epoch:", epoch)
    print("train mean_loss: ", round(total_loss / len(num), 4))
    print("train mean_acc: ", round(sum(accs) / len(accs), 4))
    train_acc_list.append(round(sum(accs) / len(accs), 4))
    tran_loss_list.append(round(total_loss / len(num), 4))
    return sum(accs) / len(accs)
def evaluate10(model, epoch, num):
    vertex_arr = np.load("../tsfm/vertex_arr_sort_svd.npy", allow_pickle=True)  # 1843
    mol_adj_arr = np.load("../tsfm/mol_adj_arr_sort_svd.npy", allow_pickle=True)
    msp_arr = np.load("../tsfm/msp_arr_sort.npy", allow_pickle=True)
    model.eval()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for batch, i in enumerate(range(0, len(num), batch_size)):
            seq_len = min(batch_size, len(num) - i)
            src = getInput0(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
            labels_graph = mol_adj_data[i:i + seq_len]
            preds = model(src,vertex_data[i:i + seq_len])  # batch, 3, 4
            preds_bond = torch.argmax(preds, dim=2)  # batch 3

            loss = criterion(preds.contiguous().view(-1, 4), labels.view(-1))
            total_loss += round(loss.item(), 4)
            acc, preds_graph = accuracy(preds_bond, labels_graph, vertex_data[i:i + seq_len])
            accs += acc
            if (epoch - 1) % 50 == 0 and batch == 0:
                print(labels_graph[0])
                print(preds_graph[0])
        print("valid mean_loss: ", round(total_loss / len(num), 4))
        print("valid mean_accs: ", round(sum(accs) / len(accs), 4))
        valid_acc_list.append(round(sum(accs) / len(accs), 4))
        valid_loss_list.append(round(total_loss / len(num), 4))
# transformer with edge
def train11(model, epoch, num):
    model.train()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num) - i)
        src = getInput11(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
        labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
        labels_graph = mol_adj_data[i:i + seq_len]
        preds = model(src)  # batch, edge_num, 4
        print(preds)
        preds_bond = torch.argmax(preds, dim=-1)  # batch edge_num

        optimizer.zero_grad()
        loss = criterion(preds.view(-1, 4), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc, preds_graph = accuracy(preds_bond, labels_graph, vertex_data[i:i + seq_len])
        accs += acc
        if (epoch - 1) % 50 == 0 and i == 0:
            print(labels_graph[0])
            print(preds_graph[0])
    print("epoch:", epoch)
    print("train mean_loss: ", round(total_loss / len(num), 4))
    print("train mean_acc: ", round(sum(accs) / len(accs), 4))
    train_acc_list.append(round(sum(accs) / len(accs), 4))
    tran_loss_list.append(round(total_loss / len(num), 4))
    return sum(accs) / len(accs)
def evaluate11(model, epoch, num):
    model.eval()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for i in range(0, len(num), batch_size):
            seq_len = min(batch_size, len(num) - i)
            src = getInput11(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
            labels_graph = mol_adj_data[i:i + seq_len]
            preds = model(src)  # batch, 3, 4
            preds_bond = torch.argmax(preds, dim=-1)  # batch 3

            loss = criterion(preds.contiguous().view(-1, 4), labels.view(-1))
            total_loss += round(loss.item(), 4)
            acc, preds_graph = accuracy(preds_bond, labels_graph, vertex_data[i:i + seq_len])
            accs += acc
            if (epoch - 1) % 50 == 0 and i == 0:
                print(labels_graph[0])
                print(preds_graph[0])
        print("valid mean_loss: ", round(total_loss / len(num), 4))
        print("valid mean_accs: ", round(sum(accs) / len(accs), 4))
        valid_acc_list.append(round(sum(accs) / len(accs), 4))
        valid_loss_list.append(round(total_loss / len(num), 4))
def train12(model, epoch, num):
    model.train()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num) - i)
        src = getInput12(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
        labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
        labels_graph = mol_adj_data[i:i + seq_len]
        preds = model(src)  # batch, edge_num, 4
        preds_bond = torch.argmax(preds, dim=-1)  # batch edge_num

        optimizer.zero_grad()
        loss = criterion(preds.view(-1, 4), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc, preds_graph = accuracy(preds_bond, labels_graph, vertex_data[i:i + seq_len])
        accs += acc
        if (epoch - 1) % 50 == 0 and i == 0:
            print(labels_graph[0])
            print(preds_graph[0])
    print("epoch:", epoch)
    print("train mean_loss: ", round(total_loss / len(num), 4))
    print("train mean_acc: ", round(sum(accs) / len(accs), 4))
    train_acc_list.append(round(sum(accs) / len(accs), 4))
    tran_loss_list.append(round(total_loss / len(num), 4))
    return sum(accs) / len(accs)
def evaluate12(model, epoch, num):
    model.eval()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for i in range(0, len(num), batch_size):
            seq_len = min(batch_size, len(num) - i)
            src = getInput12(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
            labels_graph = mol_adj_data[i:i + seq_len]
            preds = model(src)  # batch, 3, 4
            preds_bond = torch.argmax(preds, dim=-1)  # batch 3

            loss = criterion(preds.contiguous().view(-1, 4), labels.view(-1))
            total_loss += round(loss.item(), 4)
            acc, preds_graph = accuracy(preds_bond, labels_graph, vertex_data[i:i + seq_len])
            accs += acc
            if (epoch - 1) % 50 == 0 and i == 0:
                print(labels_graph[0])
                print(preds_graph[0])
        print("valid mean_loss: ", round(total_loss / len(num), 4))
        print("valid mean_accs: ", round(sum(accs) / len(accs), 4))
        valid_acc_list.append(round(sum(accs) / len(accs), 4))
        valid_loss_list.append(round(total_loss / len(num), 4))
#trainsformer with reinforcement loss
def train31(model,epoch,num):
    model.train()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num) - i)
        src = getInput11(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
        labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
        labels_graph = mol_adj_data[i:i + seq_len]
        probs = model(src)  # batch, edge_num, 4
        print(probs[0])
        #loss = criterion(probs.view(-1, 4), labels.view(-1))

        m = torch.distributions.Categorical(probs)
        #print(probs)
        preds_bond = m.sample()  # edge_num
        print(preds_bond[0])
        log_probs = []
        rewards = []
        for ii in range(len(preds_bond[0])):
            if labels[0][ii] != padding_idx:
                bond_type = preds_bond[0][ii]
                log_probs.append(torch.log(probs[0][ii][bond_type]))
                if int(preds_bond[0][ii]) == int(labels[0][ii]):
                    rewards.append(10)
                else:
                    rewards.append(1)
        ave_rewards = sum(rewards) / len(rewards)
        loss = torch.Tensor(torch.stack(log_probs)).sum()*(-ave_rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc, preds_graph = accuracy(preds_bond, labels_graph,vertex_data[i:i+seq_len])
        accs += acc
        if batch==0 and (epoch-1)%50==0:
            print(labels_graph[0])
            print(preds_graph[0])
    print("epoch:", epoch)
    print("train mean_loss: ", round(total_loss/len(num),4))
    print("train mean_acc: ", round(sum(accs)/len(accs),4))
    train_acc_list.append(round(sum(accs)/len(accs),4))
    tran_loss_list.append(round(total_loss/len(num),4))
    return sum(accs)/len(accs)
def evaluate31(model,epoch,num):
    model.eval()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for batch, i in enumerate(range(0, len(num), batch_size)):
            seq_len = min(batch_size, len(num) - i)
            src = getInput11(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
            labels_graph = mol_adj_data[i:i + seq_len]
            probs = model(src)  # batch, 3, 4
            # loss = criterion(probs.view(-1, 4), labels.view(-1))

            m = torch.distributions.Categorical(probs)
            preds_bond = m.sample()  # edge_num
            log_probs = []
            rewards = []
            for ii in range(len(preds_bond[0])):
                if labels[0][ii] != padding_idx:
                    bond_type = preds_bond[0][ii]
                    log_probs.append(torch.log(probs[0][ii][bond_type]))
                    if int(preds_bond[0][ii]) == int(labels[0][ii]):
                        rewards.append(10)
                    else:
                        rewards.append(1)
            ave_rewards = sum(rewards) / len(rewards)
            loss = torch.Tensor(torch.stack(log_probs)).sum()*(-ave_rewards)

            total_loss += loss.item()
            acc, preds_graph = accuracy(preds_bond, labels_graph, vertex_data[i:i + seq_len])
            accs += acc
            if batch == 0 and (epoch - 1) % 50 == 0:
                print(labels_graph[0])
                print(preds_graph[0])
    print("valid mean_loss: ", round(total_loss/len(num),4))
    print("valid mean_acc: ", round(sum(accs)/len(accs),4))
    valid_acc_list.append(round(sum(accs)/len(accs),4))
    valid_loss_list.append(round(total_loss/len(num),4))
    return sum(accs)/len(accs)
def train32(model,epoch,num):
    model.train()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num) - i)
        src = getInput12(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
        labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
        labels_graph = mol_adj_data[i:i + seq_len]
        probs = model(src)  # batch, edge_num, 4

        loss,preds_bond = reinforceLoss(probs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc, preds_graph = accuracy(preds_bond, labels_graph,vertex_data[i:i+seq_len])
        accs += acc
        if batch==0 and (epoch-1)%50==0:
            print(labels_graph[0])
            print(preds_graph[0])
    print("epoch:", epoch)
    print("train mean_loss: ", round(total_loss/len(num),4))
    print("train mean_acc: ", round(sum(accs)/len(accs),4))
    train_acc_list.append(round(sum(accs)/len(accs),4))
    tran_loss_list.append(round(total_loss/len(num),4))
    return sum(accs)/len(accs)
def evaluate32(model,epoch,num):
    model.eval()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for batch, i in enumerate(range(0, len(num), batch_size)):
            seq_len = min(batch_size, len(num) - i)
            src = getInput12(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
            labels_graph = mol_adj_data[i:i + seq_len]
            probs = model(src)  # batch, 3, 4

            loss, preds_bond = reinforceLoss(probs, labels)
            total_loss += loss.item()
            acc, preds_graph = accuracy(preds_bond, labels_graph, vertex_data[i:i + seq_len])
            accs += acc
            if batch == 0 and (epoch - 1) % 50 == 0:
                print(labels_graph[0])
                print(preds_graph[0])
    print("valid mean_loss: ", round(total_loss/len(num),4))
    print("valid mean_acc: ", round(sum(accs)/len(accs),4))
    valid_acc_list.append(round(sum(accs)/len(accs),4))
    valid_loss_list.append(round(total_loss/len(num),4))
    return sum(accs)/len(accs)

def plot_result(epoch):
    x1 = range(0,epoch)
    x2 = range(0,epoch)
    y1 = train_acc_list
    y2 = tran_loss_list
    y3 = valid_acc_list
    y4 = valid_loss_list
    # plt.subplot(2, 1, 1)
    # plt.plot(x1, y1, '-', label="Train_Accuracy")
    # plt.plot(x1, y3, '-', label="Valid_Accuracy")
    # plt.ylabel('Accuracy')
    # plt.legend(loc='best')
    # plt.subplot(2, 1, 2)
    # plt.plot(x2, y2, '-', label="Train_Loss")
    # plt.plot(x2, y4, '-', label="Valid_Loss")
    # plt.xlabel('epoch')
    # plt.ylabel('Loss')
    # plt.legend(loc='best')
    # plt.show()
def train_transformer(epoch, num):
    # model.load_state_dict(torch.load('model_type11.pkl'))
    for i in range(1, 1 + epoch):
        train31(model, i, num)
        evaluate31(model, i, range(1600, 1610))
    torch.save(model.state_dict(),'model_type12.pkl')
    plot_result(epoch)

train_acc_list, tran_loss_list, valid_acc_list, valid_loss_list = [],[],[],[]
train_transformer(10, num=range(160,180))

# #Testing
# model.load_state_dict(torch.load('model_type11.pkl'))
# evaluate11(model,1,range(16))
