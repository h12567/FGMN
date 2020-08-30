import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

from torch_geometric.data import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops, to_dense_adj
import torch_geometric.transforms as T
import matplotlib.pyplot as plt

from FGMN_dataset_2 import FGMNDataset
import utils

dim = 64
bond_type = 4
lr_mult=1
b=lr_mult*64
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

        nn = Sequential(Linear(1, 32), ReLU(), Linear(32, dim * dim))
        self.order = 5
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.linLast = torch.nn.Linear(dim, bond_type)

    def forward(self, data):
        # nodes = data.x
        # edge_attr = data.edge_attr_2

        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        edge_attr = data.edge_attr_2[:, :4].contiguous()
        # adj = to_dense_adj(data.edge_index_2, batch=None,
        #                    edge_attr=edge_attr.argmax(-1)+1).squeeze(0)

        # fact, fact_type = utils.get_edgeatomfactorsntypes(
        #     adj, nodes=data.x,
        #     edge_index_2=data.edge_index_2,
        #     edge_attr_2=data.edge_attr_2
        # )

        # get_factorsntypes(adj, self.order, atoms=data.x,
        #                   edge_index=data.edge_index, edge_attr=data.edge_attr)

        for k in range(3):
            m = F.relu(self.conv(out, data.edge_index_2, data.edge_attr_2))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = F.log_softmax(self.linLast(out), dim=-1)
        # edge_indicator_idx = (torch.Tensor([x[0] for x in data.x]) == EDGE_VARIABLE).nonzero()
        # return pred[edge_indicator_idx]
        # return pred
        return out

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

save_file = "models/model.pth"
torch.save(model.state_dict(), save_file)
np.save("acc_data/train_acc_list.npy", np.array(train_acc_list))
np.save("acc_data/train_loss_list.npy", np.array(train_loss_list))
a = 1
