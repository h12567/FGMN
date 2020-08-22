import sys
sys.path.append('./')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from Alchemy_dataset import TencentAlchemyDataset
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops,to_dense_adj
from datetime import datetime
import time
import logging
import pandas as pd

from torch.nn import Sequential, Linear, ReLU, GRU
from lrbp_layer_alchemy import *
import time
import argparse
from torch_geometric.nn.inits import *
import numpy as np
np.set_printoptions(precision=5, suppress=True,linewidth=np.inf)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_load', action='store_true', default=False)
parser.add_argument('--model_save', action='store_true', default=False)
parser.add_argument('--ckpt', type=str, default='')
parser.add_argument('--mode', help='mode \\in {CAT, BT, CABT, CABTA}', type=str,
                            default='CABT')
parser.add_argument('--test', action='store_true', default=False)
args = parser.parse_args()
print ('MODE :',args.mode)

dim = 64

class MyTransform(object):
    def __call__(self, data):
        data.x = torch.cat((data.x, data.pos),dim=1)
        return data

class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device
        data.edge_attr_2 = data.edge_attr.clone()
        data.edge_index_2 = data.edge_index.clone()

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([Complete(), T.Distance(norm=False), T.Center(), MyTransform()]) #   

torch.manual_seed(3)
train_dataset = TencentAlchemyDataset(root='data/alchemy',
                                mode='dev',
                                transform=transform)
train_dataset = train_dataset.shuffle()
mean = train_dataset.data.y.mean(dim=0, keepdim=True)
std = train_dataset.data.y.std(dim=0, keepdim=True)
train_dataset.data.y = (train_dataset.data.y - mean) / std

val_dataset = TencentAlchemyDataset(root='data/alchemy',
                                mode='valid',
                                transform=transform)
val_dataset.data.y = (val_dataset.data.y - mean) / std

test_dataset = TencentAlchemyDataset(root='data/alchemy',
                                mode='test',
                                transform=transform)
test_dataset.data.y = (test_dataset.data.y - mean) / std
mean, std = mean.to(device), std.to(device)

## Standardize

mn = train_dataset.data.x[:,7].mean(dim=0,keepdims=True)
st = train_dataset.data.x[:,7].std(dim=0,keepdims=True)
train_dataset.data.x[:,7] = (train_dataset.data.x[:,7] - mn) / st
val_dataset.data.x[:,7] = (val_dataset.data.x[:,7] - mn) / st
test_dataset.data.x[:,7] = (test_dataset.data.x[:,7] - mn) / st

train_dataset.data.x[:,8:11][train_dataset.data.x[:,8:11]==0] = -1.
val_dataset.data.x[:,8:11][val_dataset.data.x[:,8:11]==0] = -1.
test_dataset.data.x[:,8:11][test_dataset.data.x[:,8:11]==0] = -1.

print ('train:', len(train_dataset), 'val:', len(val_dataset), 'test:', len(test_dataset))

lr_mult=1
b=lr_mult*64
test_loader = DataLoader(test_dataset, batch_size=b, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=b, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=b, shuffle=True)


def get_factorsntypes(adj, order, atoms, edge_index, edge_attr):
    edges = []
    edge_type = []
    atom_type = []
    atom_edges = []
    degree = (adj!=0).sum(1)
    deg_sort = torch.sort(degree)
    numer = torch.arange(adj.size(0),dtype=torch.long,device=adj.device)
    for i in range(2,order):
        if i==1:
            ed = edge_index.t()
            ed_ty = edge_attr[:,:4].argmax(-1)+1
            edge_type_i = torch.cat((torch.zeros(ed_ty.size(0),dtype=torch.long,device=adj.device).unsqueeze(1), ed_ty.unsqueeze(1)),dim=1)
            edges.append(ed)
            edge_type.append(edge_type_i)
            atom_type.append(atoms[ed[:,0]])
        else:
            ind_i = deg_sort.indices[deg_sort.values==i]
            adj_i = adj[ind_i]
            val_i = adj_i.nonzero()[:,1].contiguous().view(-1,i)
            edges_i = torch.cat((ind_i.unsqueeze(1), val_i), dim=1)
            atoms_i = torch.cat((atoms[ind_i].unsqueeze(1), atoms[val_i]),dim=1)
            edge_type_i = adj_i[adj_i!=0].view(-1,i).long()
            if edge_type_i.size(0)>0:
                edge_type_i = torch.cat((torch.zeros(edge_type_i.size(0),dtype=torch.long,device=adj.device).unsqueeze(1), edge_type_i),dim=1)
                atom_type_i = atoms[ind_i]
                edges.append(edges_i)
                edge_type.append(edge_type_i)
                atom_type.append(atom_type_i)
                atom_edges.append(atoms_i)

    return edges, edge_type, atom_type, atom_edges

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin0 = torch.nn.Linear(train_dataset.num_features, dim)
        self.linear = torch.nn.Linear(train_dataset.num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(2*dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2*2 * dim, 2*2 *dim)
        self.lin2 = torch.nn.Linear(2*2 *dim, 12)

        self.order = 5
        self.bpnet1 = BPNet(num_iters=1, order=self.order, in_dim=dim, rank=512, mode=args.mode)
        self.f1 = BPNet(num_iters=1, order=self.order, in_dim=dim, rank=512, mode=args.mode)
        self.f2 = BPNet(num_iters=1, order=self.order, in_dim=dim, rank=512, mode=args.mode)
        self.f3 = BPNet(num_iters=1, order=self.order, in_dim=dim, rank=512, mode=args.mode)
        self.f_mod = torch.nn.ModuleList()
        self.f_mod.extend([self.f1, self.f2, self.f3])
        self.linear = torch.nn.Linear(dim, dim)
        self.weight = Parameter(torch.Tensor(1,dim))
        zeros(self.weight)

    def forward(self, data):
        atoms = data.x[:,:7].argmax(dim=1)
        edge_attr = data.edge_attr_2[:,:4].contiguous()
        adj = to_dense_adj(data.edge_index_2, batch=None, edge_attr=edge_attr.argmax(-1)+1).squeeze(0)
        fact, fact_type, atom_type, atom_edges = get_factorsntypes(adj, self.order, atoms=atoms,
                                    edge_index=data.edge_index, edge_attr=data.edge_attr)

        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        for k in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        conv_1 = out
        
        for k in range(3):
            isfirst = True
            for i in range(len(fact)):
                if isfirst:
                    x_1 = self.f_mod[k](out, fact[i], fact_type[i], atom_type[i], atom_edges[i]).unsqueeze(0)
                    isfirst = False
                else:
                    tmp = self.f_mod[k](out, fact[i], fact_type[i], atom_type[i], atom_edges[i]).unsqueeze(0)
                    x_1 = torch.cat((x_1,tmp),dim=0)            
            out = out + F.relu(self.linear((self.weight*x_1.sum(dim=0))))        
        conv_2 = out
        
        out = torch.cat((conv_1, conv_2),dim=1)
        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001*lr_mult)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)


def train():
    model.train()
    loss_all = 0

    lf = torch.nn.L1Loss()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = lf(model(data), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return (loss_all / len(train_loader.dataset))


@torch.no_grad()
def test(loader, istest=False):
    model.eval()
    error = torch.zeros([1, 12]).to(device)
    err = torch.zeros([1, 12]).to(device)
    for data in loader:
        data = data.to(device)
        pred = model(data)
        error += ((data.y - pred ).abs()).sum(dim=0)
    
    error = error / len(loader.dataset)
    error_log = torch.log(error)
    if istest: 
        print ('Test error: ')
        print (np.array2string(error.squeeze().cpu().numpy()))
    
    return error.mean().item(), error_log.mean().item()


results = []
results_log = []
time_log = []
best_val_error = None
start = end = 0.
if args.model_load:
    print ("loading model..")
    model.load_state_dict(torch.load(args.ckpt))

if args.test:
    print ('started testing.. ')
    test_error, log_test_error = test(test_loader, istest=True)
    print('Test MAE: {:.7f}'.format(test_error))
else:
    print ("started training..")
    test_error = None
    log_test_error = None
    best_val_error = None
    for epoch in range(1, 201):
        start = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train()
        val_error, log_val_error = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error, log_test_error = test(test_loader,True)
            best_val_error = val_error
            if args.model_save:
                torch.save(model.state_dict(),"alchemy/models/Alchemy_alltargets_"+ args.mode +".pth")

        end = time.time()
        time_log.append(end-start)
        print('Epoch: {:03d},Time: {:3f}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
              'Test MAE: {:.7f}, Log Test MAE: {:.7f}'.format(epoch, (end-start), lr, loss, val_error, test_error, log_test_error))

        if lr < 0.000001:
            print("Converged.")
            break

        results.append(test_error)
        results_log.append(log_test_error)

    print("########################")
    results = np.array(results)
    print(results.mean(), results.std())

    time_log = np.array(time_log)
    print(time_log.mean(), time_log.std())

