from Models import Encoder, EdgeClassify, EncoderEdgeClassify
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import time
from getInput import GetInput
from pre_knowledge import generate_pre_knowledge_adj_mat
import os
import copy

max_atoms = 13
atom_type=4
msp_len = 800
src_vocab = max(1000, max_atoms)
d_model = 18 # size of atom embedding after encorder
# Note that for now d_model must be 18 because we don't really do embedding yet.
heads =  6 # number of heads in multi-head attention
N = 5 # number of loops of encoderLayer
dropout = 0.1
batch_size = 32
num_bonds_prediction = 4 # (no_bond, single, double, triple)

model = EncoderEdgeClassify(src_vocab, d_model, N, heads, dropout, msp_len, max_atoms, num_bonds_prediction)
vertex_arr = np.load("../transformer/vertex_arr_sort_svd.npy", allow_pickle=True) #225
mol_adj_arr = np.load("../transformer/mol_adj_arr_sort_svd.npy", allow_pickle=True)
msp_arr = np.load("../nist_db_helpers/msp_arr.npy", allow_pickle=True)

weights = [1/119, 1/20,1/16,1/10] #[ 1 / number of instances for each class]
class_weights = torch.FloatTensor(weights)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
new_inputs = GetInput(vertex_arr, msp_arr, max_atoms, atom_type, k=30, type=4)
new_inputs = new_inputs.detach().numpy()

func_group_knowledge = [
    (0, [(2, 1), (2, 2)]), # this means atom type 0 connects to atom type 2 with bond 1 and atom type 2 with bond 2
]

num_extra_nodes = new_inputs.shape[1] - max_atoms
all_pre_distance_matrix = []
for i, E in enumerate(mol_adj_arr):
    new_E = generate_pre_knowledge_adj_mat(vertex_arr[i], num_extra_nodes,
                                           E, func_group_knowledge)
    all_pre_distance_matrix.append(new_E)
all_pre_distance_matrix = np.array(all_pre_distance_matrix)

def train_model(model,train_idx):
    model.train()
    src_data = new_inputs[train_idx]
    label_data = mol_adj_arr[train_idx]
    pre_distance_matrix = all_pre_distance_matrix[train_idx]

    old_model_state_dict = None
    for batch, i in enumerate(range(0, len(src_data), batch_size)):
        optimizer.zero_grad()
        #get data and labels
        seq_len = min(batch_size, len(src_data)- i)
        src = torch.from_numpy(src_data[i:i+seq_len])
        sub_pre_distance_matrix = torch.from_numpy(pre_distance_matrix[i:i+seq_len]).long()
        # print(src[0][0])
        # exit(0)
        labels = torch.from_numpy(label_data[i:i+seq_len]).long() #[batch, max-atom,max-atom]=[8,13,13]
        src_mask = None
        preds = model(src, src_mask, max_atoms, sub_pre_distance_matrix) #[8,13,13,4] batch-size, outwidth, outheight, out-channel
        # print(preds.shape,labels.shape)
        # print(torch.argmax(preds[0], dim=2))
        # print(labels[0])
        accuracy_arr = []
        for j in range(len(src)):
            num_equal = (labels[j] == torch.argmax(preds[j], dim=2)).sum().float()
            accuracy_arr.append(num_equal / labels[j].numel())
        acc = np.mean(accuracy_arr)
        loss = criterion(preds.view(-1,num_bonds_prediction), labels.view(-1))
        loss.backward()
        optimizer.step()
        print("WEIGHT")
        print(model.state_dict()["encoder.layers.0.attn.self_attention_weight"])
        print(model.state_dict()["encoder.layers.0.attn.distance_mat_weight"])
        for i in model.state_dict():
            if old_model_state_dict:
                if (model.state_dict()[i] != old_model_state_dict[i]).sum() > 0:
                    if i in ["encoder.layers.0.attn.self_attention_weight",
                             "encoder.layers.0.attn.distance_mat_weight"]:
                        a = 1
        old_model_state_dict = copy.deepcopy(model.state_dict())
        # print(model.state_dict()["encoder.layers.0.attn.q_linear.weight"])
        print("batch:{:1d}/{:d}, loss:{:3f}, acc:{:3f}".format(batch,len(src_data)//batch_size,
                                                               loss.item(), acc))
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
    pre_distance_matrix = all_pre_distance_matrix[test_idx]

    batch_size = len(src_data)
    with torch.no_grad():
        for i in range(0, len(src_data), batch_size):
            print(i)
            seq_len = min(batch_size, len(src_data) - i)
            src = torch.from_numpy(src_data[i:i + seq_len])
            sub_pre_distance_matrix = torch.from_numpy(pre_distance_matrix[i:i + seq_len]).long()
            labels = torch.from_numpy(label_data[i:i + seq_len]).long()  # [batch, max-atom,max-atom]=[8,13,13]
            src_mask = None
            preds = model(src, src_mask, max_atoms, sub_pre_distance_matrix)
            # for j in range(len(labels)):
            #     print(labels[j])
            #     print(torch.argmax(preds[j],dim=2))
            loss = criterion(preds.view(-1, num_bonds_prediction), labels.view(-1))
            accuracy_arr = []
            for j in range(len(src)):
                num_equal = (labels[j] == torch.argmax(preds[j], dim=2)).sum().float()
                accuracy_arr.append(num_equal / labels[j].numel())
            acc = np.mean(accuracy_arr)
            print("TEST LOSS \n")
            print("test loss:{:3f}, acc:{:3f}".format(loss.item(), acc))

def transformer(epoch):
    model_filename = "../model_store/model_with_pre_knowledge_type_4"
    for i in range(1,1+epoch):
        if (os.path.isfile(model_filename)):
            model.load_state_dict(torch.load(model_filename))
        print("epoch: ",i)
        train_model(model, range(0,175))
        #evaluate(model, range(120, 121))
        if i % 30 == 0:
            test(model, range(175, 225))
            torch.save(model.state_dict(), model_filename)
    # torch.save(model.state_dict(), 'modeltsfm.pt')
    # test(model,range(187, 190))

transformer(60000)
