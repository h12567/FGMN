import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F

class HardConstraintNet(nn.Module):
    pass

class HighOrderNet(nn.Module):
    def __init__(self, order=3, in_dim=512, out_dim=512, hidden_dim=None, mode='CABT'):
        super(HighOrderNet, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = self.in_dim if hidden_dim is None else hidden_dim
        self.out_dim = out_dim
        self.order = order
        self.mods = nn.ModuleList()
        for i in range(self.order):
            self.mods.append(nn.Linear(self.in_dim, self.hidden_dim, bias=True))
        self.fc = nn.Linear(self.hidden_dim, out_dim, bias=True)
        self.linear_list = nn.ModuleList()

        self.max_num_atoms = 13
        self.num_params = self.max_num_atoms ** 2

        self.params = Parameter(torch.FloatTensor(self.num_params, self.hidden_dim, self.out_dim))
        self.bias = Parameter(torch.FloatTensor(self.num_params, 1, self.out_dim))

    def forward(self, x, fact, inp, msg_to, order):
        isfirst = True
        for i in range(order):
            if i == msg_to:
                continue
            if isfirst:
                fact_val = ((inp[i]).unsqueeze(-1))
                isfirst = False
            else:
                fact_val = torch.cat((fact_val, ((inp[i])).unsqueeze(-1)), -1)
        fact_prod = fact_val.prod(-1)

        atom_pairs = x[fact.clone()[:, 0]][:, 1:].clone()
        ids = atom_pairs[:, 0] * self.max_num_atoms + atom_pairs[:, 1]
        ids = ids.clone()

        weights = self.params.view(self.num_params, self.hidden_dim * self.out_dim)[ids.long()].contiguous()
        bias = self.bias[ids.long()]
        params_cat = weights.view(-1, self.hidden_dim, self.out_dim)

        transform = (fact_prod.unsqueeze(1) @ params_cat) + bias
        return transform.squeeze(1)

# Note that FGNet only flows messages to atom node and msp nodes, from type B and
# type C factors
# so when calculating nodes_to_edges, separating out based on dimension before calculating
# the messages
class FGNet(nn.Module):
    def __init__(self, num_iters=3, order=3, in_dim=13, rank=128, num_classes=26, fact_type=None):
        super(FGNet, self).__init__()
        self.rank = rank
        self.latent_dim = 64
        self.bond_type = 4
        self.max_num_atoms = 13
        if fact_type not in ["A", "B", "C"]:
            raise Exception("Must specify factor type correctly")
        self.fact_type = fact_type

        if self.fact_type == "B":
            self.num_params = self.max_num_atoms**2

        self.highorder_func = nn.ModuleList()
        self.max_order = 12
        for i in range(self.max_order):
            self.highorder_func.append(HighOrderNet(
                order=2,
                in_dim=self.latent_dim, out_dim=self.latent_dim,
                hidden_dim=self.rank,
            ))

        self.params = Parameter(torch.FloatTensor(self.num_params, self.latent_dim, self.rank))
        self.bias = Parameter(torch.FloatTensor(self.num_params, 1, self.rank))


    def forward(self, x, nodes, fact, fact_dim, fact_type=None):
        numer = torch.arange(0, fact.size(0), dtype=torch.long, device=nodes.device)

        # if fact_type == "A":
        #     sub_fact = fact
        # elif fact_type == "B":
        #     sub_fact = fact[:, 1:]
        # else:
        #     sub_fact = fact

        sub_fact = fact

        nodes_to_edges = []

        transform_lst = []
        for i in range(sub_fact.size(1)):
            nodes_to_edges1 = nodes.data.new(nodes.size(0), sub_fact.size(0)).zero_()
            nodes_to_edges1.view(-1)[sub_fact[:, i] * sub_fact.size(0) + numer] = 1
            nodes_to_edges.append(nodes_to_edges1)

            if self.fact_type == "B":
                atom_pairs = x[fact.clone()[:, 0]][:, 1:].clone()
                ids = atom_pairs[:, 0] * self.max_num_atoms + atom_pairs[:, 1]
                ids = ids.clone()

            rnodes = nodes[sub_fact[:, i]]

            weights = self.params.view(self.num_params, self.latent_dim * self.rank)[ids.long()].contiguous()
            params = weights.view(-1, self.latent_dim, self.rank)
            bias = self.bias[ids.long()]

            transform = (rnodes.unsqueeze(1) @ params) + bias
            transform_lst.append(transform.squeeze(1).unsqueeze(0))

        # (num_nodes_per_factor, num_factors, rank)
        transform_cat = F.relu(torch.cat(transform_lst, dim=0))
        inp_x = transform_cat

        for i in range(sub_fact.size(1)):
            # self.highorder_func[i](x, fact, inp=inp_x, msg_to=i, order=sub_fact.size(1))
            # current_msg_x = torch.zeros(sub_fact.size(0), self.latent_dim)
            current_msg_x = self.highorder_func[i](x, fact, inp=inp_x, msg_to=i, order=sub_fact.size(1))
            if i == 0:
                node_msg = (nodes_to_edges[i]) @ current_msg_x
            else:
                node_msg += (nodes_to_edges[i]) @ current_msg_x

        return node_msg
