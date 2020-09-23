import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Sequential, Linear, ReLU, GRU

from torch_geometric.data import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops, to_dense_adj
import torch_geometric.transforms as T
import matplotlib.pyplot as plt

from FGMN_dataset_2 import FGMNDataset
from fgmn_layer import FGNet, ValenceNet
import utils

dim = 16
bond_type = 4
lr_mult=1
b=lr_mult*1
num_epoches=100
NUM_MSP_PEAKS = 16
ATOM_VARIABLE = 1
EDGE_VARIABLE = 2
MSP_VARIABLE = 3
EDGE_FACTOR = 4
MSP_FACTOR = 5

class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device
        data.edge_attr_2 = data.edge_attr.clone()
        data.edge_index_2 = data.edge_index.clone()

        # row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        # col = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        #
        # row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        # col = col.repeat(data.num_nodes)
        # edge_index = torch.stack([row, col], dim=0)
        #
        # edge_attr = None
        # if data.edge_attr is not None:
        #     idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
        #     size = list(data.edge_attr.size())
        #     size[0] = data.num_nodes * data.num_nodes
        #     edge_attr = data.edge_attr.new_zeros(size)
        #     edge_attr[idx] = data.edge_attr
        #
        # edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        # data.edge_attr = edge_attr
        # data.edge_index = edge_index

        return data

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin0 = torch.nn.Linear(dataset.num_features, dim)
        self.lin1 = torch.nn.Linear(dim, bond_type)
        self.linUp = torch.nn.Linear(bond_type, dim)
        self.linDown = torch.nn.Linear(dim, bond_type)

        nn = Sequential(Linear(1, 32), ReLU(), Linear(32, dim * dim))
        self.order = 5
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.linLast = torch.nn.Linear(dim, bond_type)

        self.fB1 = FGNet(num_iters=1, in_dim=dim, rank=512, fact_type="B")
        self.fB2 = FGNet(num_iters=1, in_dim=dim, rank=512, fact_type="B")
        self.fB3 = FGNet(num_iters=1, in_dim=dim, rank=512, fact_type="B")
        self.fB4 = FGNet(num_iters=1, in_dim=dim, rank=512, fact_type="B")
        self.f_mod_B = torch.nn.ModuleList()
        self.f_mod_B.extend([self.fB1, self.fB2, self.fB3, self.fB4])

        self.fC1 = FGNet(num_iters=1, in_dim=dim, rank=512, fact_type="C")
        self.fC2 = FGNet(num_iters=1, in_dim=dim, rank=512, fact_type="C")
        self.fC3 = FGNet(num_iters=1, in_dim=dim, rank=512, fact_type="C")
        self.fC4 = FGNet(num_iters=1, in_dim=dim, rank=512, fact_type="C")
        self.f_mod_C = torch.nn.ModuleList()
        self.f_mod_C.extend([self.fC1, self.fC2, self.fC3, self.fC4])

        self.fA1 = FGNet(num_iters=1, in_dim=bond_type, rank=512, fact_type="A")
        self.fA2 = FGNet(num_iters=1, in_dim=bond_type, rank=512, fact_type="A")
        self.fA3 = FGNet(num_iters=1, in_dim=bond_type, rank=512, fact_type="A")
        self.fA4 = FGNet(num_iters=1, in_dim=bond_type, rank=512, fact_type="A")
        self.f_mod_A = torch.nn.ModuleList()
        self.f_mod_A.extend([self.fA1, self.fA2, self.fA3, self.fA4])

        self.fA_valence = ValenceNet()

        self.weight = Parameter(torch.Tensor(1, dim))
        self.linear = torch.nn.Linear(dim, dim)
        self.weight_edges = Parameter(torch.Tensor(1, bond_type))
        self.linear_edges = torch.nn.Linear(bond_type, bond_type)

    def forward(self, data):
        # nodes = data.x
        # edge_attr = data.edge_attr_2

        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        edge_attr = data.edge_attr_2[:, :4].contiguous()
        adj = to_dense_adj(data.edge_index_2, batch=None,
                           edge_attr=edge_attr.argmax(-1)+1).squeeze(0)

        fact_l_B = utils.get_edgeatomfactorsntypes(
            adj, dim, bond_type,
            nodes=data.x,
            edge_index_2=data.edge_index_2,
            edge_attr_2=data.edge_attr_2,
        )

        fact_l_C = utils.get_mspatomfactorsntypes(
            adj, dim, bond_type,
            nodes=data.x,
            edge_index_2=data.edge_index_2,
            edge_attr_2=data.edge_attr_2,
        )

        fact_l_A = utils.get_edgesedgesfactorsnttypes(
            adj, dim, bond_type,
            nodes=data.x,
            edge_index_2=data.edge_index_2,
            edge_attr_2=data.edge_attr_2,
        )

        for k in range(3):
            m = F.relu(self.conv(out, data.edge_index_2, data.edge_attr_2))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out_edges = F.relu(self.lin1(out))# (num_nodes, bond_type)
        # out_edges = torch.zeros(out.shape[0], bond_type).cuda()

        # conv_1 = out

        for k in range(1):
            ################## PREPARING OUT_COMBINE ##########################################################
            edge_variable_idxes = torch.flatten((data.x[:, 0] == EDGE_VARIABLE).nonzero())
            out_edges_upscale = self.linUp(out_edges)
            out_combine = out.clone()
            out_combine[edge_variable_idxes] = out_edges_upscale[edge_variable_idxes]
            ##################################################################################################
            all_msg, all_msg_to_edge_A, all_msg_to_edge_B = None, None, None
            for i in range(len(fact_l_B)): #this length is always 1
                msg = self.f_mod_B[k](data.x, out_combine, fact_l_B[i], fact_type="B").unsqueeze(0)
                msg_to_edge = self.linDown(msg)
                all_msg, all_msg_to_edge_B = msg, msg_to_edge

            for i in range(len(fact_l_C)):
                if fact_l_C[i] is not None:
                    msg = self.f_mod_C[k](data.x, out_combine, fact_l_C[i], fact_type="C").unsqueeze(0)
                    all_msg = torch.cat([all_msg, msg])

            for i in range(len(fact_l_A)):
                if fact_l_A[i] is not None:
                    # msg_to_edge = self.f_mod_A[k](data.x, out_edges, fact_l_A[i], fact_type="A").unsqueeze(0)
                    # all_msg_to_edge = torch.cat([all_msg_to_edge, msg_to_edge])
                    msg_to_edge = self.fA_valence.compute(
                        data.x, out_edges, fact_l_A[i]
                    )
                    msg_to_edge = msg_to_edge.cuda()
                    if all_msg_to_edge_A is not None:
                        all_msg_to_edge_A = torch.cat([all_msg_to_edge_A, msg_to_edge])
                    else:
                        all_msg_to_edge_A = msg_to_edge


            combine_edge_msg = (all_msg_to_edge_A.sum(dim=0) * all_msg_to_edge_B.sum(dim=0))
            out = out + F.relu(self.linear((self.weight * all_msg.sum(dim=0))))
            out_edges = out_edges + F.relu(self.linear_edges((self.weight_edges * combine_edge_msg)))


        # out = F.log_softmax(self.linLast(out), dim=-1
        out_edges_final = F.log_softmax(out_edges, dim=-1)
        return out_edges_final

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(3)

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'FGMN')
transform = T.Compose([Complete()])
dataset = FGMNDataset(path, transform=transform).shuffle()

train_loader = DataLoader(dataset, batch_size=b, shuffle=False)

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001*lr_mult)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=5,
                                                       min_lr=0.00001)

def accuracy(pred, target):
    total_match = 0
    total_zeros = 0
    for i in range(len(pred)):
        if pred[i][0].item() == target[i][0].item():
            total_match += 1
        if target[i][0].item() == 0:
            total_zeros += 1
    return total_match / len(pred), total_zeros / len(pred)

def train():
    model.train()
    loss_all, acc_all, base_acc_all = 0, 0, 0
    # lf = torch.nn.L1Loss()
    lf = torch.nn.NLLLoss()
    torch.autograd.set_detect_anomaly(True)
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # node_input_idx = torch.flatten((data.x[:, 0] <= 3).nonzero())
        # node_data = data.x[node_input_idx]
        # factor_idx = torch.flatten((data.x[:, 0] > 3).nonzero())
        # factor_data = data.x[factor_idx]

        edge_indicator_idx = (torch.Tensor([x[0] for x in data.x]) == EDGE_VARIABLE).nonzero()
        out = model(data)
        # pred = torch.argmax(out, dim=1)
        optimizer.zero_grad()
        valid_labels = data.y[edge_indicator_idx]
        valid_out = out[edge_indicator_idx]
        # loss = lf(out[edge_indicator_idx], data.y[edge_indicator_idx])
        loss = lf(
            valid_out.view(-1, 4),
            valid_labels.view(-1)
        )
        # loss = lf(model(data), data.y[edge_indicator_idx])
        # loss = lf(pred, data.y)
        loss.backward()
        print(data)
        print(str(loss) + "\n")
        valid_pred = torch.argmax(valid_out, dim=-1)
        loss_all += loss.item() * data.num_graphs
        acc_all += accuracy(valid_pred, valid_labels)[0] * data.num_graphs
        base_acc_all += accuracy(valid_pred, valid_labels)[1] * data.num_graphs
        optimizer.step()
    print("BASE ACCURACY " + str(base_acc_all / len(train_loader.dataset)) + "\n")
    return (loss_all / len(train_loader.dataset)), (acc_all / len(train_loader.dataset))

train_acc_list, train_loss_list = [], []


def plot_result(num_epoches):
    x1 = range(0, num_epoches)
    x2 = range(0, num_epoches)
    y1 = train_acc_list
    y2 = train_loss_list
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, '-', label="Train_Accuracy")
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '-', label="Train_Loss")
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()

print("Started training")
lr = scheduler.optimizer.param_groups[0]['lr']
for i in range(1, 1 + num_epoches):
    print("Start epoch " + str(i))
    loss, acc = train()
    train_loss_list.append(loss)
    train_acc_list.append(acc)
plot_result(num_epoches)

save_file = "models/model2.pth"
torch.save(model.state_dict(), save_file)
np.save("acc_data/train_acc_list2.npy", np.array(train_acc_list))
np.save("acc_data/train_loss_list2.npy", np.array(train_loss_list))
a = 1
