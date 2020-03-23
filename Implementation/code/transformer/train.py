from Models import Encoder, EdgeClassify, EncoderEdgeClassify
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time

max_atoms = 13
msp_len = 800
src_vocab = max(1000, max_atoms)
d_model = 512 # size of atom embedding after encorder
heads = 8 # number of heads in multi-head attention
N = 3 # number of loops of encoderLayer
dropout = 0.1
batch_size = 8
num_bonds_prediction = 4 # (no_bond, single, double, triple)

model = EncoderEdgeClassify(src_vocab, d_model, N, heads, dropout, msp_len, max_atoms, num_bonds_prediction)
vertex_arr = np.load("../transformer/vertex_arr_2.npy", allow_pickle=True) #225
mol_adj_arr = np.load("../transformer/mol_adj_arr_2.npy", allow_pickle=True)
msp_arr = np.load("../transformer/peaks_arr_10.npy", allow_pickle=True)
new_inputs = np.zeros((len(vertex_arr),max_atoms,8)) #8 means 7peaks, 1 atom

for i in range(len(vertex_arr)):
    for j in range(len(vertex_arr[i])):
        if vertex_arr[i][j]==0:
            new_inputs[i,j,0]=12
            #new_inputs[i,j,1:5]=[0,0,0,0]
        elif vertex_arr[i][j]==1:
            new_inputs[i,j,0]=1
            #new_inputs[i,j,1:5]=[1,0,0,0]
        elif vertex_arr[i][j]==2:
            new_inputs[i,j,0]=16
            #new_inputs[i,j,1:5]=[0,0,1,0]
        elif vertex_arr[i][j]==3:
            new_inputs[i,j,0]=14
            #new_inputs[i,j,1:5]=[0,0,0,1]
        for p in range(7):
            new_inputs[i,j,1+p]=msp_arr[i][p][1]


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train_model(model,train_idx):
    model.train()
    src_data = new_inputs[train_idx]
    label_data = mol_adj_arr[train_idx]

    for batch, i in enumerate(range(0, len(src_data), batch_size)):
        #get data and labels
        seq_len = min(batch_size, len(src_data)- i)
        src = torch.from_numpy(src_data[i:i+seq_len].astype(int))
        labels = torch.from_numpy(label_data[i:i+seq_len]).long() #[batch, max-atom,max-atom]=[8,13,13]
        optimizer.zero_grad()
        src_mask = None
        preds = model(src, src_mask, max_atoms) #[8,13,13,4] batch-size, outwidth, outheight, out-channel
        #print(labels[0])
        #print(preds[0])
        print(torch.argmax(preds[0], dim=2)) # the prediction matrix
        loss = criterion(preds.view(-1, num_bonds_prediction), labels.view(-1))
        loss.backward()
        optimizer.step()
        print("batch:{:1d}/{:d}, loss:{:3f}".format(batch,len(src_data)//batch_size,loss.item()))

def evaluate(model,test_idx):
    print("eval!")
    model.eval()
    src_data = new_inputs[test_idx]
    label_data = mol_adj_arr[test_idx]
    with torch.no_grad():
        for i in range(0, len(src_data), batch_size):
            print(i)
            seq_len = min(batch_size, len(src_data) - i)
            src = torch.from_numpy(src_data[i:i + seq_len].astype(int))
            labels = torch.from_numpy(label_data[i:i + seq_len]).long()  # [batch, max-atom,max-atom]=[8,13,13]
            src_mask = None
            preds = model(src, src_mask, max_atoms)
            # print(labels[0])
            # print(torch.argmax(preds[0],dim=2))
            loss = criterion(preds.view(-1, num_bonds_prediction), labels.view(-1))
            print("loss: ", loss.item())
def test(model,test_idx):
    print("Test!")
    model.eval()
    src_data = new_inputs[test_idx]
    label_data = mol_adj_arr[test_idx]
    with torch.no_grad():
        for i in range(0, len(src_data), batch_size):
            seq_len = min(batch_size, len(src_data) - i)
            src = torch.from_numpy(src_data[i:i + seq_len].astype(int))
            labels = torch.from_numpy(label_data[i:i +  seq_len]).long()  # [batch, max-atom,max-atom]=[8,13,13]
            src_mask = None
            preds = model(src, src_mask, max_atoms)
            # for j in range(len(labels)):
            #     print(labels[j])
            #     print(torch.argmax(preds[j],dim=2))
            loss = criterion(preds.view(-1, num_bonds_prediction), labels.view(-1))
            print("loss: ", loss.item())

def accuracy(labels, outputs, num):
    acc = []
    for i in range(num):
        l=labels[i]
        y=outputs[i]
        count=0
        for j in range(max_atoms):
            for jj in range(max_atoms):
                if l[j,jj]==y[j,jj]: count = count+1
        acc.append(count)
    return acc


def transformer(epoch):
    for i in range(1,1+epoch):
        print("epoch: ",i)
        train_model(model, range(10))
        #evaluate(model, range(55, 57))
    #torch.save(model.state_dict(),'model1.pt')
    #test(model,range(177, 180))

transformer(300)

