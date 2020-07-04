import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

vertex_arr = np.load("../tsfm/vertex_arr_sort_per.npy", allow_pickle=True) #1843
mol_adj_arr = np.load("../tsfm/mol_adj_arr_sort_per.npy", allow_pickle=True)
msp_arr = np.load("../tsfm/msp_arr_sort_per.npy", allow_pickle=True)
H_num = np.load("../tsfm/h_num_per.npy", allow_pickle=True)
msp_len = 800
k=20
atom_type=2
padding_idx = 799 #must > 0 and < msp_len
dropout = 0.3
batch_size = 1
atom_mass = [12,1,16,14] #[12,1,16]
atomic_number = [[1,0,0],[0,0,0],[0,1,0],[0,0,1]] #[C, H, O, N}
bond_number = [4,1,2] #[C, H, O]
default_valence = [[1,1,1],[1,0,0],[1,1,0],[1,1,1]]
atom_pos_mass = [28,29,33,40,32+12,16*3,12+4,24+1,24+2,24+3,28+2,28+3,12*3,36+1,38,24+16,41,41,42,48,49,36+16,52,53,28*2,56,57,12*5]

edge_num = 78 #3#29*15#78 #3
d_model=256
max_atoms = 13 # 3
max_len0 = 300
max_len1 = 3 * edge_num + k

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx)#.unsqueeze(-2)
def get_pad_mask0(seq,vertex):
    mask =  torch.ByteTensor([[[False]*4]*edge_num]*seq.size(0)) #torch.zeros((seq.size(0),seq.size(1),4))
    for b in range(seq.size(0)):
        length = len(vertex[b])
        idx = 0
        for i in range(max_atoms):
            for j in range(i+1,max_atoms):
                if i >= length or j >= length : mask[b,idx] = torch.ByteTensor([0,1,1,1])
                idx += 1
    return mask
def get_pad_mask1(seq):
    mask =  torch.ByteTensor([[[False]*4]*edge_num]*seq.size(0)) #torch.zeros((seq.size(0),seq.size(1),4))
    for b in range(seq.size(0)):
        for i in range(edge_num):
            if seq[b,i*3]==padding_idx: mask[b,i] = torch.ByteTensor([0,1,1,1])
            if seq[b, i * 3] == 16 or seq[b, i * 3+1] == 16 : mask[b, i] = torch.ByteTensor([0, 0, 0, 1])
    return mask

class Classify0(nn.Module): #linear
    def __init__(self,padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(msp_len,d_model,padding_idx)
        self.ll1 = nn.Linear(d_model, edge_num)
        self.ll2 = nn.Linear(k,4)
    def forward(self,src,vertex):
        self.mask = get_pad_mask0(src,vertex)  # [batch, edge_num,4]
        output = self.embedding(src) #[batch, k, d_model]
        output = self.ll1(output) #[batch, k, edge_num]
        output = output.permute(0, 2, 1) #[batch, edge_num, k]
        output = self.ll2(output) #[batch, edge_num, 4]
        output = output.masked_fill(self.mask, -1e9)
        output = F.log_softmax(output,dim=2)
        return output #[batch, edge_num=3, bond=4]
class Classify1(nn.Module):
    def __init__(self, padding_idx):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(msp_len, d_model, padding_idx)
        self.ll1 = nn.Linear(d_model, edge_num)
        self.ll2 = nn.Linear(max_len1, 4)

    def forward(self, src):
        self.mask = get_pad_mask1(src)  # [batch, edge_num,4]
        output = self.embedding(src)  # [batch, max_len=45, d_model]
        output = self.ll1(output)  # [batch, max_len=45, edge_num=3]
        output = output.permute(0, 2, 1)  # [batch, edge_num, max_len = 45]
        output = self.ll2(output)  # [batch, edge_num=3, bond=4]
        output = output.masked_fill(self.mask, -1e9)
        output = F.log_softmax(output, dim=2)
        return output  # [batch, edge_num=3, bond=4]
class Classify20(nn.Module): #linear
    def __init__(self,padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(msp_len,d_model,padding_idx)
        self.ll1 = nn.Linear(d_model, edge_num)
        self.ll2 = nn.Linear(k,4)
    def forward(self,src,vertex):
        self.mask = get_pad_mask0(src,vertex)  # [batch, edge_num,4]
        output = self.embedding(src) #[batch, k, d_model]
        output = self.ll1(output) #[batch, k, edge_num]
        output = output.permute(0, 2, 1) #[batch, edge_num, k]
        output = self.ll2(output) #[batch, edge_num, 4]
        output = output.masked_fill(self.mask, -1e9)
        output = F.softmax(output,dim=-1)
        return output #[batch, edge_num, bond=4]
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
    # [A1-weight, A2-weight, 1 1 0,A1, A3,1 1 1  A2, A3,1 0 0 msp1[x], ...,mspk[x]] 45
    src = torch.LongTensor([[padding_idx] * max_len1] * len(vertex))  # [batch, 45]
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
                if idx == max_len1: break
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

#model
model = Classify20(padding_idx)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.NLLLoss(ignore_index=padding_idx) #CrossEntropyLoss()

#linear svd
def train0(model,epoch,num):
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
        seq_len = min(batch_size, len(num)- i)
        src = getInput0(vertex_data[i:i+seq_len],msp_arr_data[i:i+seq_len])
        labels = getLabel(mol_adj_data[i:i+seq_len],vertex_data[i:i+seq_len])
        labels_graph = torch.Tensor(mol_adj_data[i:i+seq_len])
        preds = model(src,vertex_data[i:i+seq_len]) #batch, 3, 4
        preds_bond = torch.argmax(preds, dim=2) #batch 3

        optimizer.zero_grad()
        loss = criterion(preds.view(-1, 4), labels.view(-1))
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
def evaluate0(model,epoch,num):
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
            labels = getLabel(mol_adj_data[i:i + seq_len],vertex_data[i:i+seq_len])
            label_graph = mol_adj_data[i:i + seq_len]
            preds = model(src,vertex_data[i:i + seq_len])  # batch, 3, 4
            preds_bond = torch.argmax(preds, dim=2)  # batch 3

            loss = criterion(preds.contiguous().view(-1, 4), labels.view(-1))
            total_loss += loss.item()
            acc, preds_graph = accuracy(preds_bond, label_graph,vertex_data[i:i+seq_len])
            accs += acc
            if (epoch-1) % 50 == 0 and batch==0:
                print(label_graph[0])
                print(preds_graph[0])
        print("valid mean_loss: ", round(total_loss / len(num), 4))
        print("valid mean_accs: ", round(sum(accs)/len(accs),4))
        valid_acc_list.append(round(sum(accs)/len(accs),4))
        valid_loss_list.append(round(total_loss / len(num), 4))
#linear edge
def train1(model, epoch, num):
    model.train()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []

    for batch, i in enumerate(range(0, len(num), batch_size)):
        seq_len = min(batch_size, len(num) - i)
        src = getInput1(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
        labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
        labels_graph = torch.Tensor(mol_adj_data[i:i + seq_len])
        preds = model(src)  # batch, 3, 4
        preds_bond = torch.argmax(preds, dim=2)  # batch 3

        optimizer.zero_grad()
        loss = criterion(preds.view(-1, 4), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc, preds_graph = accuracy(preds_bond, labels_graph, vertex_data[i:i + seq_len])
        accs += acc
        if batch == 0 and (epoch - 1) % 50 == 0:
            print(labels_graph[0])
            print(preds_graph[0])
    print("epoch:", epoch)
    print("train mean_loss: ", round(total_loss / len(num), 4))
    print("train mean_acc: ", round(sum(accs) / len(accs), 4))
    train_acc_list.append(round(sum(accs) / len(accs), 4))
    tran_loss_list.append(round(total_loss / len(num), 4))
def evaluate1(model, epoch, num):
    model.eval()
    vertex_data = vertex_arr[num]
    mol_adj_data = mol_adj_arr[num]
    msp_arr_data = msp_arr[num]
    total_loss = 0
    accs = []
    with torch.no_grad():
        for batch, i in enumerate(range(0, len(num), batch_size)):
            seq_len = min(batch_size, len(num) - i)
            src = getInput1(vertex_data[i:i + seq_len], msp_arr_data[i:i + seq_len])
            labels = getLabel(mol_adj_data[i:i + seq_len], vertex_data[i:i + seq_len])
            labels_graph = mol_adj_data[i:i + seq_len]
            preds = model(src)  # batch, 3, 4
            preds_bond = torch.argmax(preds, dim=2)  # batch 3

            loss = criterion(preds.contiguous().view(-1, 4), labels.view(-1))
            total_loss += loss.item()
            acc, preds_graph = accuracy(preds_bond, labels_graph, vertex_data[i:i + seq_len])
            accs += acc
            if (epoch - 1) % 50 == 0 and batch == 0:
                print(labels_graph[0])
                print(preds_graph[0])
        print("valid mean_loss: ", round(total_loss / len(num), 4))
        print("valid mean_accs: ", round(sum(accs) / len(accs), 4))
        valid_acc_list.append(round(sum(accs) / len(accs), 4))
        valid_loss_list.append(round(total_loss / len(num), 4))
#linear reinforce
def train20(model,epoch,num):
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
        seq_len = min(batch_size, len(num)- i)
        src = getInput0(vertex_data[i:i+seq_len],msp_arr_data[i:i+seq_len])
        labels = getLabel(mol_adj_data[i:i+seq_len],vertex_data[i:i+seq_len])
        labels_graph = torch.Tensor(mol_adj_data[i:i+seq_len])
        probs = model(src,vertex_data[i:i+seq_len]) #batch, edge_num, 4
        #loss = criterion(probs.view(-1, 4), labels.view(-1))

        m = torch.distributions.Categorical(probs)
        #print(probs)
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
def evaluate20(model,epoch,num):
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
            labels_graph = torch.Tensor(mol_adj_data[i:i + seq_len])
            probs = model(src, vertex_data[i:i + seq_len])  # batch, 3, 4
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
def plot_result(epoch):
    x1 = range(0,epoch)
    x2 = range(0,epoch)
    y1 = train_acc_list
    y2 = tran_loss_list
    y3 = valid_acc_list
    y4 = valid_loss_list
    plt.subplot(2, 1, 1)
    # plt.plot(x1, y1, 'o-',color='r')
    plt.plot(x1, y1, '-', label="Train_Accuracy")
    plt.plot(x1, y3, '-', label="Valid_Accuracy")
    #plt.title("Accuracy")
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '-', label="Train_Loss")
    plt.plot(x2, y4, '-', label="Valid_Loss")
    #plt.title("Loss")
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()
def train_linear(epoch,num):
    model.load_state_dict(torch.load('model_type0.pkl'))
    for i in range(1,1+epoch):
        train20(model,i,num)
        evaluate20(model, i,range(1600,1610))
    #torch.save(model.state_dict(),'model_type20.pkl')
    plot_result(epoch)

train_acc_list, tran_loss_list, valid_acc_list, valid_loss_list = [],[],[],[]
train_linear(60,num=range(20))

# #Testing
# model.load_state_dict(torch.load('model_type1.pkl'))
# evaluate1(model,1,range(1600, 1610))
