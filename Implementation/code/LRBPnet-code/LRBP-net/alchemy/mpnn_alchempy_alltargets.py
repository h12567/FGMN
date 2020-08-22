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
parser.add_argument('--test', action='store_true', default=False)
args = parser.parse_args()

dim = 64

class MyTransform(object):
    def __call__(self, data):
        data.x = torch.cat((data.x, data.pos),dim=1)
        return data

class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

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


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin0 = torch.nn.Linear(train_dataset.num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 12)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([Complete(), T.Distance(norm=False)]) #Complete(), 
torch.manual_seed(3)
train_dataset = TencentAlchemyDataset(root='data/alchemy',
                                mode='dev',
                                transform=transform)
train_dataset = train_dataset.shuffle()
val_dataset = TencentAlchemyDataset(root='data/alchemy',
                                mode='valid',
                                transform=transform)
test_dataset = TencentAlchemyDataset(root='data/alchemy',
                                mode='test',
                                transform=transform)

y = torch.cat((train_dataset.data.y, val_dataset.data.y, test_dataset.data.y),dim=0)
mean = y.mean(dim=0, keepdim=True)
std = y.std(dim=0, keepdim=True)
train_dataset.data.y = (train_dataset.data.y - mean) / std
val_dataset.data.y = (val_dataset.data.y - mean) / std
test_dataset.data.y = (test_dataset.data.y - mean) / std

mean, std = mean.to(device), std.to(device)

print ('train:', len(train_dataset), 'val:', len(val_dataset), 'test:', len(test_dataset))

lr_mult=1
b=lr_mult*64
test_loader = DataLoader(test_dataset, batch_size=b, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=b, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=b, shuffle=True)


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001*lr_mult)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)

def train():
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y)
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

def load_state_dict(model, ckpt):
    model_dict = model.state_dict()
    ckpt_dict = torch.load(ckpt)
    pretrained_dict = {k:v for k,v in ckpt_dict.items() \
        if k in model_dict and v.size()==model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return

results = []
results_log = []
time_log = []
best_val_error = None
start = end = 0.
if args.model_load:
    print ("loading model..")
    load_state_dict(model,args.ckpt)
    print ("done loading ckpt")

if args.test:
    print ("testing model..")
    test_error, log_test_error = test(test_loader,istest=True)
    print('Test MAE: {:.7f}, Log Test MAE: {:.7f}'.format(test_error, log_test_error))
else:
    print ("Started training..")
    test_error = None
    log_test_error = None
    best_val_error = None
    for epoch in range(1, 21):
        start = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train()
        val_error, log_val_error = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error, log_test_error = test(test_loader,True)
            best_val_error = val_error
            if args.model_save:
                torch.save(model.state_dict(),"alchemy/models/Alchemy_MPNN_alltargets.pth")

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


