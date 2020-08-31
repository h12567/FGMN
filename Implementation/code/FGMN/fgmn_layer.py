import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F

class HardConstraintNet(nn.Module):
    pass

class HighOrderNet(nn.Module):
    pass

# Note that FGNet only flows messages to atom node and msp nodes, from type B and
# type C factors
# so when calculating nodes_to_edges, separating out based on dimension before calculating
# the messages
class FGNetTypeB(nn.Module):
    def __init__(self, num_iters=3, order=3, in_dim=13, rank=128, num_classes=26):
        super(FGNetTypeB, self).__init__()
        self.rank = rank
        self.latent_dim = 64
        self.bond_type = 4
        self.max_num_atoms = 13
        self.num_params = self.max_num_atoms**2

        self.higherorder_func = nn.ModuleList()
        for i in range(2):
            self.higherorder_func.append(HighOrderNet())

        self.params = Parameter(torch.FloatTensor(self.num_params, self.latent_dim, self.rank))
        self.bias = Parameter(torch.FloatTensor(self.num_params, 1, self.rank))

    def forward(self, x, nodes, fact, fact_dim, fact_type=None):
        numer = torch.arange(0, fact.size(0), dtype=torch.long, device=nodes.device)
        transform_cat = torch.zeros(fact.size(1), fact.size(0), self.rank)

        if fact_type not in ["A", "B", "C"]:
            raise Exception("Must specify factor type correctly")

        if fact_type == "A":
            sub_fact = fact
        elif fact_type == "B":
            sub_fact = fact[:, 1:]
        else:
            sub_fact = fact

        nodes_to_edges = []

        transform_lst = []
        for i in range(sub_fact.size(1)):
            nodes_to_edges1 = nodes.data.new(nodes.size(0), sub_fact.size(0)).zero_()
            nodes_to_edges1.view(-1)[sub_fact[:, i] * sub_fact.size(0) + numer] = 1
            nodes_to_edges.append(nodes_to_edges1)

            atom_pairs = x[fact[:, 0]][:, 1:]
            ids = atom_pairs[:, 0] * self.max_num_atoms + atom_pairs[:, 1]

            rnodes = nodes[sub_fact[:, i]]

            weights = self.params.view(self.num_params, self.latent_dim * self.rank)[ids.long()].contiguous()
            params = weights.view(-1, self.latent_dim, self.rank)
            bias = self.bias[ids.long()]

            transform = (rnodes.unsqueeze(1) @ params) + bias
            transform_lst.append(transform.squeeze(1).unsqueeze(0))

        transform_cat = F.relu(torch.cat(transform_lst, dim=0))
        inp_x = transform_cat

        for i in range(sub_fact.size(1)):
            current_msg_x = torch.zeros(sub_fact.size(0), self.latent_dim)
            if i == 0:
                node_msg = (nodes_to_edges[i]) @ current_msg_x
            else:
                node_msg += (nodes_to_edges[i]) @ current_msg_x
