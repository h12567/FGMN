from transformer.Models import Encoder, EdgeClassify, EncoderEdgeClassify
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
from transformer.getInput import GetInput

max_atoms = 13
atom_type=4
msp_len = 800
src_vocab = max(1000, max_atoms)
d_model = 18 # size of atom embedding after encorder
heads =  3# number of heads in multi-head attention
N = 3 # number of loops of encoderLayer
dropout = 0.1
batch_size = 8
num_bonds_prediction = 4 # (no_bond, single, double, triple)

model = EncoderEdgeClassify(src_vocab, d_model, N, heads, dropout, msp_len, max_atoms, num_bonds_prediction)
vertex_arr = np.load("../nist_db_helpers/vertex_arr.npy", allow_pickle=True) #225
mol_adj_arr = np.load("../nist_db_helpers/mol_adj_arr.npy", allow_pickle=True)
msp_arr = np.load("../nist_db_helpers/msp_arr.npy", allow_pickle=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
new_inputs = GetInput(vertex_arr, msp_arr, max_atoms, atom_type, k=30, type=4)
new_inputs = new_inputs.detach().numpy()


def train_model(model,train_idx):
    model.train()
    src_data = new_inputs[train_idx]
    label_data = mol_adj_arr[train_idx]

    for batch, i in enumerate(range(0, len(src_data), batch_size)):
        optimizer.zero_grad()
        #get data and labels
        seq_len = min(batch_size, len(src_data)- i)
        src = torch.from_numpy(src_data[i:i+seq_len])
        # print(src[0][0])
        # exit(0)
        labels = torch.from_numpy(label_data[i:i+seq_len]).long() #[batch, max-atom,max-atom]=[8,13,13]
        src_mask = None
        preds = model(src, src_mask, max_atoms) #[8,13,13,4] batch-size, outwidth, outheight, out-channel
        print(preds.shape,labels.shape)
        print(torch.argmax(preds[0], dim=2))
        loss = criterion(preds.view(-1,num_bonds_prediction), labels.view(-1))
        loss.backward()
        optimizer.step()
        print("batch:{:1d}/{:d}, loss:{:3f}".format(batch,len(src_data)//batch_size,loss.item()))
    #torch.save(model.state_dict(),'model1.pkl')

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
            loss = criterion(preds.view(-1, num_bonds_prediction), labels.view(-1))
            print("loss: ", loss.item())

def test(model,test_idx):
    print("Test!")
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
            for j in range(len(labels)):
                print(labels[j])
                print(torch.argmax(preds[j],dim=2))
            loss = criterion(preds.view(-1, num_bonds_prediction), labels.view(-1))
            print("loss: ", loss.item())

def transformer(epoch):
    for i in range(1,1+epoch):
        print("epoch: ",i)
        train_model(model, range(100,116))
        #evaluate(model, range(120, 121))
    # torch.save(model.state_dict(), 'modeltsfm.pt')
    # test(model,range(187, 190))


transformer(16)

