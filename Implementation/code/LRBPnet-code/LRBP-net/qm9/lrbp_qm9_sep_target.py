import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops,to_dense_adj
import time
import argparse
import sys
sys.path.append('./')
from lrbp_layer_qm9 import *
from torch_geometric.nn.inits import *
import numpy as np
np.set_printoptions(precision=5, suppress=True,linewidth=np.inf)

parser = argparse.ArgumentParser()
parser.add_argument('--target', default=0)
parser.add_argument('--model_load', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--ckpt', type=str, default='')
parser.add_argument('--mode', help='mode \\in {CAT, BT, CABT, CABTA}', type=str,
                            default='CABT')
parser.add_argument('--model_save', action='store_true', default=False)

args = parser.parse_args()
target = int(args.target)
print('Target: {}'.format(target))
print('MODE  : {}'.format(args.mode))


dim = 64


HAR2EV = 27.2113825435
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])

class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        data.x = torch.cat((data.x, data.pos),dim=1)
        data.y = data.y[:, target]
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
        self.lin0 = torch.nn.Linear(dataset.num_features, dim)
        self.linear = torch.nn.Linear(dataset.num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(2*dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2*2 * dim, 2*2 *dim)
        self.lin2 = torch.nn.Linear(2*2 *dim, 1)

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
        atoms = data.x[:,:5].argmax(dim=1)
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
        return out.view(-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(3)
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
transform = T.Compose([Complete(), T.Distance(), T.Center(),MyTransform()])
dataset = QM9(path, transform=transform).shuffle()

# Normalize targets to mean = 0 and std = 1.
dataset.data.y = dataset.data.y / conversion.view(1,-1)
dataset.data.y = dataset.data.y[:,0:12]
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
#mean, std = mean.to(device), std.to(device)
mean, std = mean[:, target].item(), std[:, target].item()

# Split datasets.
tenpercent = int(len(dataset) * 0.1)
#Standardize input
norm_ids = torch.LongTensor([5,12])#13,14,15
mn = dataset.data.x[tenpercent:][:,norm_ids].mean(dim=0,keepdims=True)
st = dataset.data.x[tenpercent:][:,norm_ids].std(dim=0,keepdims=True)
dataset.data.x[:,norm_ids] = (dataset.data.x[:,norm_ids] - mn) / st
dataset.data.x[:,6:9][dataset.data.x[:,6:9]==0] = -1.

test_dataset = dataset[:tenpercent]
val_dataset = dataset[tenpercent:2*tenpercent]
train_dataset = dataset[2*tenpercent:]
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
    error = 0.
    for data in loader:
        data = data.to(device)
        pred = model(data)
        error += (data.y * std - pred * std).abs().sum().item()
    err = error / len(loader.dataset)
    return err

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
best_val_error = None
start = end = 0.
if args.model_load:
    print ("loading model..")
    load_state_dict(model,args.ckpt)
    print ("done loading ckpt")
    
if args.test:
    print ("Testing..")
    test_error = test(test_loader,istest=True)
    print('Test MAE: {:.7f}'.format(test_error))

else:
    print ("Started training..")
    test_error = None
    log_test_error = None
    best_val_error = None
    for epoch in range(1, 201):
        start = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train()
        val_error = test(val_loader)
        scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            test_error = test(test_loader,True)
            best_val_error = val_error
            if args.model_save :
                save_file= "qm9/models/qm9_"+ args.mode+"_target_"+args.target +".pth"
                torch.save(model.state_dict(),save_file)

        end = time.time()
        print('Epoch: {:03d},Time: {:3f}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
              'Test MAE: {:.7f}'.format(epoch, (end-start), lr, loss, val_error, test_error))

        if lr < 0.000001:
            print("Converged.")
            break

    results.append(test_error)
    
    print("########################")
    print(results)
    results = np.array(results)
    print(results.mean(), results.std())
