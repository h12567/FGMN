import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import copy
from tsfm.Layers import EncoderLayer #,DecoderLayer
from tsfm.Sublayers import Norm
from tsfm.Embed import PositionalEncoder
import matplotlib.pyplot as plt
import tsfm.getInput as getInput

vertex_arr = np.load("../tsfm/vertex_arr_test.npy", allow_pickle=True) #1843
mol_adj_arr = np.load("../tsfm/mol_adj_arr_test.npy", allow_pickle=True)
msp_arr = np.load("../tsfm/msp_arr_sort_per.npy", allow_pickle=True)

msp_len = 800
k = 20
atom_type = 2
padding_idx = 799  # must > 0 and < msp_len
dropout = 0.2
batch_size = 1
atom_mass = [12, 1, 16, 14]  # [12,1,16]
atomic_number = [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]]  # [C, H, O, N}
bond_number = [4, 1, 2]  # [C, H, O]
default_valence = [[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]]

edge_num = 78  # 3#29*15#78 #3
d_model = 256
max_atoms = 13  # 3
max_len11 = 15 * edge_num + k
max_len12 = 2 * edge_num + k

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

def get_pad_mask11(seq):
    mask =  torch.ByteTensor([[[False]*4]*edge_num]*seq.size(0)) #torch.zeros((seq.size(0),seq.size(1),4))
    for b in range(seq.size(0)):
        for i in range(edge_num):
            if seq[b,i*15]==padding_idx: mask[b,i] = torch.ByteTensor([0,1,1,1])
            if seq[b, i * 15] == 16 or seq[b, i * 15+1] == 16 : mask[b, i] = torch.ByteTensor([0, 0, 0, 1])
    return mask
def get_pad_mask13(seq):
    mask =  torch.ByteTensor([[False]*4]*seq.size(0)) #torch.zeros((seq.size(0),seq.size(1),4))
    for b in range(seq.size(0)):
        if seq[b,0] == 16 or seq[b, 0] == 16 :
            mask[0] = torch.ByteTensor([0,0,0,1])
    return mask
# transformer with edge non-pos
class Classify11(nn.Module):
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
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, src):
        self.mask = get_pad_mask11(src)  # [batch, edge_num,4]
        mask = get_pad_mask(src, self.padding_idx).view(src.size(0), -1).unsqueeze(-1)
        output = self.embedding(src)  # [batch, k, d_model=512]
        for i in range(self.N):
            output = self.layers[i](output, mask)
        output = self.ll1(output)  # [batch, max_len2, edge_num]
       # output = self.dropout1(output)
        output = output.permute(0, 2, 1)  # [batch, edge_num, max_len2]
        output = self.ll2(output)  # [batch, edge_num, 4]
        output = output.masked_fill(self.mask,-1e9)
        output = F.log_softmax(output, dim=-1)
        return output  # [batch, edge_num=3, bond=4]
# transformer with edge pos
class Classify12(nn.Module):
    def __init__(self, padding_idx):
        super().__init__()
        heads = 4
        self.N = 1
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len, d_model, self.padding_idx)
        self.pe = PositionalEncoder(d_model=d_model, max_seq_len=edge_num * 15, dropout=0.2)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout=0.15), self.N)
        self.norm = Norm(d_model)
        self.ll1 = nn.Linear(d_model, edge_num)
        self.ll2 = nn.Linear(max_len11, 4)

    def forward(self, src):
        self.mask = get_pad_mask11(src)  # [batch, edge_num,4]
        mask = get_pad_mask(src, self.padding_idx).view(src.size(0), -1).unsqueeze(-1)
        output = self.embedding(src)  # [batch, k, d_model=512]
        output = torch.cat((self.pe(output[:, :15 * edge_num]), output[:, 15 * edge_num:]), dim=1)
        for i in range(self.N):
            output = self.layers[i](output, mask)
        output = self.ll1(output)  # [batch, max_len12, edge_num]
        output = output.permute(0, 2, 1)  # [batch, edge_num, max_len12]
        output = self.ll2(output)  # [batch, edge_num, 4]
        output = output.masked_fill(self.mask,-1e9)
        output = F.log_softmax(output, dim=-1)
        return output  # [batch, edge_num=3, bond=4]
#with decoder
# class Classify13(nn.Module):
#     def __init__(self, padding_idx):
#         super().__init__()
#         heads = 4
#         self.N = 1
#         self.padding_idx = padding_idx
#         self.embedding = nn.Embedding(msp_len, d_model, self.padding_idx)
#         self.enc_layers = get_clones(EncoderLayer(d_model, heads, dropout), self.N)
#         self.dec_layers = get_clones(DecoderLayer(d_model, heads, dropout), self.N)
#         self.norm = Norm(d_model)
#         #self.ll = nn.Linear(d_model*4, 4)
#     def forward(self, src, dec_input,next_dec,enc_output=None):
#         self.mask = get_pad_mask13(next_dec)  # [batch, edge_num,4]
#         dec_input = dec_input.view(src.size(0),-1)
#         enc_mask = get_pad_mask(src, self.padding_idx).view(src.size(0), -1).unsqueeze(-1)
#         if enc_output is None:
#             enc_output = self.embedding(src)  # [batch, k, d_model=512]
#             for i in range(self.N):
#                 enc_output = self.enc_layers[i](enc_output,enc_mask)
#         dec_input = self.embedding(dec_input)
#         for i in range(self.N):
#             output = self.dec_layers[i](enc_output,dec_input,None,None)
#         output = output.view(src.size(0),-1)
#         ll = nn.Linear(d_model * len(dec_input[0]), 4)
#         output = ll(output)
#         # output = output.masked_fill(self.mask, -1e9)
#         # output = F.log_softmax(output, dim=-1)
#         return output,enc_output  # [batch, edge_num=3, bond=4]

def getInput11(vertex, msp):
    # [A1-weight, A2-weight, 1 1 0,A1, A3,1 1 1  A2, A3,1 0 0 msp1[x], ...,mspk[x]] 45 = k + edge_num* *
    src = torch.LongTensor([[padding_idx] * max_len11] * len(vertex))  # [batch, 45]
    for b in range(len(vertex)):
        idx1 = 0
        for i in range(max_atoms):
            for ii in range(i + 1, max_atoms):
                if i < len(vertex[b]) and ii < len(vertex[b]):
                    src[b, idx1 * 15], src[b, idx1 * 15 + 1] = atom_mass[int(vertex[b][i])], atom_mass[int(vertex[b][ii])]
                    src[b, idx1 * 15: idx1 * 15 + len(vertex[b])] = torch.LongTensor([int(jj==i or jj==ii) for jj in range(len(vertex[b]))])
                idx1 += 1
        idx = 15 * edge_num
        for j in range(1, len(msp[b]) - 1):
            if msp[b][j - 1] < msp[b][j] and msp[b][j + 1] < msp[b][j]:
                src[b, idx] = j
                idx += 1
                if idx == max_len11: break
    return src
#with decoder
def getInput13(vertex, msp): #transformer without positional encoding
    src = torch.LongTensor([[padding_idx] * k] * len(vertex))  # [batch, k]
    dec_input = torch.LongTensor([[padding_idx] * 4*edge_num] * len(vertex)) #[bt,edge_num*4]
    for b in range(len(vertex)):
        idx = 0
        for j in range(1, len(msp[b]) - 1):
            if msp[b][j - 1] < msp[b][j] and msp[b][j + 1] < msp[b][j]:
                src[b, idx] = j
                idx += 1
                if idx == k: break
        idx1 = 0
        for i in range(max_atoms):
            for ii in range(i + 1, max_atoms):
                if i < len(vertex[b]) and ii < len(vertex[b]):
                    dec_input[b, idx1*4+0], dec_input[b, idx1*4+1] = atom_mass[int(vertex[b][i])], atom_mass[int(vertex[b][ii])]
                    dec_input[b, idx1*4+2] = idx1 + 1
                    if idx1 == 0: dec_input[b, idx1*4+3] = 2
                    elif idx1 == 1: dec_input[b, idx1*4+3] = 1
                idx1 += 1
    return src,dec_input

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
                preds_graph[b,i,j] = preds_bond[b][idx]
                preds_graph[b, j, i] = preds_graph[b,i,j]
                idx +=1
                if i < length and j < length and [i,j] not in [[0,1],[0,2]]:
                    count += 1
                    if preds_graph[b,i,j]==label_graph[b,i,j]:acc+=1
        accs.append(round(acc/(count+np.finfo(np.float32).eps),4))
    return accs, preds_graph
# model
# model = Classify12(padding_idx)
model = Classify11(padding_idx)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
criterion = nn.NLLLoss(ignore_index=padding_idx)  # CrossEntropyLoss()

def isValid(ori_bonds,pred_bond,vertex):
    bonds = ori_bonds + [pred_bond]
    preds_graph = torch.zeros((max_atoms, max_atoms))  # batch, max_atom=3, max_atom=3
    idx = 0
    for i in range(max_atoms):
        for j in range(i + 1, max_atoms):
            preds_graph[i, j] = bonds[idx]
            preds_graph[j, i] = preds_graph[i, j]
            idx += 1
            if idx >= len(bonds): break
        if idx >= len(bonds): break
    if i >= len(vertex) or j>= len(vertex): return 0
    sum_row = preds_graph[i].sum()
    sum_col = preds_graph[:,j].sum()
    max_row = bond_number[int(vertex[i])]
    max_col = bond_number[int(vertex[j])]
    if (max_row-sum_row) >= 1 and (max_col-sum_col) >= 1: return 1
    if (max_row - sum_row) >= 2 and (max_col - sum_col) >= 2: return 2
    if (max_row - sum_row) >= 3 and (max_col - sum_col) >= 3: return 3
    if (max_row - sum_row) >= 4 and (max_col - sum_col) >= 4: return 3
    else: return -1
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
            #labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
            labels_graph = mol_adj_data[i:i + seq_len]
            preds = model(src)  # batch, 3, 4
            preds_bond = torch.argmax(preds, dim=-1)  # batch 3

            #loss = criterion(preds.contiguous().view(-1, 4), labels.view(-1))
            atom_lists = getInput.find_permutation(vertex_data[i])
            losses = []
            for al in atom_lists:
                new_E = getInput.getGraph(labels_graph[0], al)
                labels = getLabel(new_E)
                loss = criterion(preds.view(-1, 4), labels.view(-1))
                losses.append(loss)
            loss = min(losses)
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
def test11(model, epoch, num):
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
        print("test mean_loss: ", round(total_loss / len(num), 4))
        print("test mean_accs: ", round(sum(accs) / len(accs), 4))
        test_acc_list.append(round(sum(accs) / len(accs), 4))
        test_loss_list.append(round(total_loss / len(num), 4))
def train12(model, epoch, num):
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
            src = getInput11(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            #labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
            labels_graph = mol_adj_data[i:i + seq_len]
            preds = model(src)  # batch, 3, 4
            preds_bond = torch.argmax(preds, dim=-1)  # batch 3

            #loss = criterion(preds.contiguous().view(-1, 4), labels.view(-1))
            atom_lists = getInput.find_permutation(vertex_data[i])
            losses = []
            for al in atom_lists:
                new_E = getInput.getGraph(labels_graph[0], al)
                labels = getLabel(new_E)
                loss = criterion(preds.view(-1, 4), labels.view(-1))
                losses.append(loss)
            loss = min(losses)
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
def test12(model, epoch, num):
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
        print("test mean_loss: ", round(total_loss / len(num), 4))
        print("test mean_accs: ", round(sum(accs) / len(accs), 4))
        test_acc_list.append(round(sum(accs) / len(accs), 4))
        test_loss_list.append(round(total_loss / len(num), 4))
# transformer with decoder edge
def train13(model, epoch, num):
    model.train()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num) - i)
        src,dec_input = getInput13(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
        labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
        labels_graph = mol_adj_data[i:i + seq_len]
        bonds= []
        bonds.append(dec_input[0,3])
        bonds.append(dec_input[0,7])
        probs = []
        for dd in range(2,edge_num):
            if dd == 2:
                preds,enc_output = model(src, dec_input[:, :(dd + 1) * 4], dec_input[:, (dd - 1) * 4:dd * 4])  # batch, 4
            else:
                preds,enc_output = model(src, dec_input[:, :(dd + 1) * 4], dec_input[:, (dd - 1) * 4:dd * 4],enc_output)
            mask = get_pad_mask13(dec_input[:, (dd - 1) * 4:dd * 4]) #[batch, edge_num,4]
            preds2 = preds.masked_fill(mask, -1e9)
            preds2 = F.log_softmax(preds2, dim=-1)
            preds_bond = torch.argmax(preds2, dim=-1)  # batch edge_num
            flag = isValid(bonds, preds_bond, vertex_data[i])
            while (flag < 1):  # illegle
                if flag <= 0:
                    mask[0] = torch.ByteTensor([0,1,1,1])
                    preds2 = preds.masked_fill(mask, -1e9)
                    preds2 = F.log_softmax(preds2, dim=-1)
                    preds_bond = torch.argmax(preds2, dim=-1)
                    labels[0, dd] = padding_idx
                    break
                elif flag == 1:
                    mask[0] = torch.ByteTensor([0,0,1,1])
                elif flag == 2:
                    mask[0] = torch.ByteTensor([0,0,0,1])
                preds2 = preds.masked_fill(mask, -1e9)
                preds2 = F.log_softmax(preds2, dim=-1)
                preds_bond = torch.argmax(preds2, dim=-1)
                flag = isValid(bonds, preds_bond, vertex_data[i])
            bonds.append(preds_bond[0])
            probs.append(preds2)
        #print(probs)
        probs = torch.stack(probs)
        optimizer.zero_grad()
        loss = criterion(probs.view(-1, 4), labels[0,2:].view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        acc, preds_graph = accuracy([torch.stack(bonds)], labels_graph, vertex_data[i:i + seq_len])
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
def evaluate13(model, epoch, num):
    model.eval()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num) - i)
        src, dec_input = getInput13(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
        labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
        labels_graph = mol_adj_data[i:i + seq_len]
        bonds = []
        bonds.append(dec_input[0, 3])
        bonds.append(dec_input[0, 7])
        probs = []
        for dd in range(2, edge_num):
            if dd == 2:
                preds, enc_output = model(src, dec_input[:, :(dd + 1) * 4],
                                          dec_input[:, (dd - 1) * 4:dd * 4])  # batch, 4
            else:
                preds, enc_output = model(src, dec_input[:, :(dd + 1) * 4], dec_input[:, (dd - 1) * 4:dd * 4],
                                          enc_output)
            mask = get_pad_mask13(dec_input[:, (dd - 1) * 4:dd * 4])  # [batch, edge_num,4]
            preds2 = preds.masked_fill(mask, -1e9)
            preds2 = F.log_softmax(preds2, dim=-1)
            preds_bond = torch.argmax(preds2, dim=-1)  # batch edge_num
            flag = isValid(bonds, preds_bond, vertex_data[i])
            while (flag < 1):  # illegle
                if flag <= 0:
                    mask[0] = torch.ByteTensor([0, 1, 1, 1])
                    preds2 = preds.masked_fill(mask, -1e9)
                    preds2 = F.log_softmax(preds2, dim=-1)
                    preds_bond = torch.argmax(preds2, dim=-1)
                    labels[0, dd] = padding_idx
                    break
                elif flag == 1:
                    mask[0] = torch.ByteTensor([0, 0, 1, 1])
                elif flag == 2:
                    mask[0] = torch.ByteTensor([0, 0, 0, 1])
                preds2 = preds.masked_fill(mask, -1e9)
                preds2 = F.log_softmax(preds2, dim=-1)
                preds_bond = torch.argmax(preds2, dim=-1)
                flag = isValid(bonds, preds_bond, vertex_data[i])
            bonds.append(preds_bond[0])
            probs.append(preds2)
        # print(probs)
        probs = torch.stack(probs)
        optimizer.zero_grad()
        loss = criterion(probs.view(-1, 4), labels[0, 2:].view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        acc, preds_graph = accuracy([torch.stack(bonds)], labels_graph, vertex_data[i:i + seq_len])
        accs += acc
        if (epoch - 1) % 50 == 0 and i == 0:
            print(labels_graph[0])
            print(preds_graph[0])
    print("valid mean_loss: ", round(total_loss / len(num), 4))
    print("valid mean_accs: ", round(sum(accs) / len(accs), 4))
    valid_acc_list.append(round(sum(accs) / len(accs), 4))
    valid_loss_list.append(round(total_loss / len(num), 4))
def test13(model, epoch, num):
    model.eval()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num) - i)
        src, dec_input = getInput13(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
        labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
        labels_graph = mol_adj_data[i:i + seq_len]
        bonds = []
        bonds.append(dec_input[0, 3])
        bonds.append(dec_input[0, 7])
        probs = []
        for dd in range(2, edge_num):
            if dd == 2:
                preds, enc_output = model(src, dec_input[:, :(dd + 1) * 4],
                                          dec_input[:, (dd - 1) * 4:dd * 4])  # batch, 4
            else:
                preds, enc_output = model(src, dec_input[:, :(dd + 1) * 4], dec_input[:, (dd - 1) * 4:dd * 4],
                                          enc_output)
            mask = get_pad_mask13(dec_input[:, (dd - 1) * 4:dd * 4])  # [batch, edge_num,4]
            preds2 = preds.masked_fill(mask, -1e9)
            preds2 = F.log_softmax(preds2, dim=-1)
            preds_bond = torch.argmax(preds2, dim=-1)  # batch edge_num
            flag = isValid(bonds, preds_bond, vertex_data[i])
            while (flag < 1):  # illegle
                if flag <= 0:
                    mask[0] = torch.ByteTensor([0, 1, 1, 1])
                    preds2 = preds.masked_fill(mask, -1e9)
                    preds2 = F.log_softmax(preds2, dim=-1)
                    preds_bond = torch.argmax(preds2, dim=-1)
                    labels[0, dd] = padding_idx
                    break
                elif flag == 1:
                    mask[0] = torch.ByteTensor([0, 0, 1, 1])
                elif flag == 2:
                    mask[0] = torch.ByteTensor([0, 0, 0, 1])
                preds2 = preds.masked_fill(mask, -1e9)
                preds2 = F.log_softmax(preds2, dim=-1)
                preds_bond = torch.argmax(preds2, dim=-1)
                flag = isValid(bonds, preds_bond, vertex_data[i])
            bonds.append(preds_bond[0])
            probs.append(preds2)
        # print(probs)
        probs = torch.stack(probs)
        optimizer.zero_grad()
        loss = criterion(probs.view(-1, 4), labels[0, 2:].view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        acc, preds_graph = accuracy([torch.stack(bonds)], labels_graph, vertex_data[i:i + seq_len])
        accs += acc
        if (epoch - 1) % 50 == 0 and i == 0:
            print(labels_graph[0])
            print(preds_graph[0])
    print("test mean_loss: ", round(total_loss / len(num), 4))
    print("test mean_accs: ", round(sum(accs) / len(accs), 4))
    test_acc_list.append(round(sum(accs) / len(accs), 4))
    test_loss_list.append(round(total_loss / len(num), 4))


def plot_result(epoch):
    x1 = range(0,epoch)
    x2 = range(0,epoch)
    y1 = train_acc_list
    y2 = tran_loss_list
    y3 = valid_acc_list
    y4 = valid_loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, '-', label="Train_Accuracy")
    plt.plot(x1, y3, '-', label="Valid_Accuracy")
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '-', label="Train_Loss")
    plt.plot(x2, y4, '-', label="Valid_Loss")
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()
def train_transformer(epoch, num):
    for i in range(1, 1 + epoch):
        train11(model, i, num)
        evaluate11(model, i, range(1500, 1700))
        test11(model, i, range(1700, 1800))
    torch.save(model.state_dict(),'model_type11.pkl')
    plot_result(epoch)

train_acc_list, tran_loss_list, valid_acc_list, valid_loss_list, test_acc_list, test_loss_list = [],[],[],[], [], []
train_transformer(200,num=range(1500))

