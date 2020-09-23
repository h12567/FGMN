import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F

class ValenceNet():
    def __init__(self):
        # super(ValenceNet, self).__init__()
        self.atom_valence_mapping = {
            0: 4, #Carbon
            1: 1, #Hydrogen
            2: 2, #Oxygen
            3: 3, #Nitrogen
        }

    def compute(self, x, nodes, fact):
        # nodes = nodes.clone()
        # nodes: (num_nodes, 4)

        atom_idxes, edge_idxes = fact[:, 0], fact[:, 1:]
        atoms_nodes, edges = x[atom_idxes], nodes[edge_idxes]
        atom_types = atoms_nodes[:, 1].tolist()
        valences = [self.atom_valence_mapping[i] for i in atom_types]
        valences = torch.tensor(valences)
        # edges: (num_factors, num_neighbors, bond_type)

        max_valence = 4
        num_factors, num_edges_per_factor, max_bond_type = edges.size(0), edges.size(1), edges.size(2) - 1
        node_msg = torch.zeros(1, x.size(0), max_bond_type+1).cuda()

        for msg_to in range(0, edges.size(1)):
            # msg_to = edges.size(1) - 1

            edges_clone = edges.clone()
            temp = edges_clone[:, -1, :].clone()
            edges_clone[:, -1, :] = edges_clone[:, msg_to, :]
            edges_clone[:, msg_to, :] = temp
            msg_to_temp = edges.size(1) - 1

            msgs = torch.zeros(num_factors, num_edges_per_factor, max_valence + 1, max_bond_type + 1).cuda()
            # msgs = msgs.clone()
            # temp_msg = torch.ones([num_factors, max_valence + 1, max_bond_type + 1]).cuda()
            temp_msg = torch.zeros([num_factors, max_valence + 1, max_bond_type + 1]).cuda()
            for i in range(4):
                temp_msg[:, i, i] = torch.ones([num_factors])
            # for i in range(0, msg_to+1):
            for i in range(0, msg_to_temp+1): #don't pl
                # temp_msg: (num_factors, max_bond_type)
                for j in range(max_valence + 1): # j is current valence REMAINING
                    max_possible_bond_type = min(j, max_bond_type)
                    # temp_msg = torch.tensor([[1] * (max_possible_bond_type + 1)] * num_factors)
                    # sub_temp_msg = temp_msg[:, :(max_possible_bond_type + 1), :]
                    sub_temp_msg = temp_msg[:, (j - max_possible_bond_type):(j+1), :]
                    new_info = edges_clone[:, i, :(max_possible_bond_type+1)]
                    new_info = torch.flip(new_info, [1])
                    new_info = torch.unsqueeze(new_info, 2)
                    new_info = F.normalize(new_info, p=2, dim=2) # Normalize g
                    msgs = msgs.clone()
                    msgs[:, i, j, :] = torch.sum(sub_temp_msg * new_info, dim=1) #.clone()
                temp_msg = msgs[:, i, :, :]

            msg_recipient = fact[:, msg_to+1] # need to plus one because there is one extra node at begin
            # msg_last = msgs[:, msg_to, :, :] #(num_factor, max_valence+1, max_bond_type)
            # msg_last = msgs[:, -1, :, :]
            msg_last = msgs[:, -2, :, :] #.clone() #last column is the msg_to node itself, we should no use
            for i, node_idx in enumerate(msg_recipient):
                node_msg[0, node_idx, :] += msg_last[i, valences[i], :]

            del msgs, temp_msg

        return node_msg

class HighOrderNet(nn.Module):
    def __init__(self, in_dim=512, out_dim=512, hidden_dim=None, fact_type=None):
        super(HighOrderNet, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = self.in_dim if hidden_dim is None else hidden_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(self.hidden_dim, out_dim, bias=True)
        self.linear_list = nn.ModuleList()

        self.max_num_atoms = 13
        self.max_msp_index = 1000
        self.fact_type = fact_type

        if self.fact_type == "B":
            self.num_params = self.max_num_atoms**2
        elif self.fact_type == "C":
            self.num_params = self.max_msp_index
        elif self.fact_type == "A":
            self.num_params = self.max_num_atoms

        self.params = Parameter(torch.FloatTensor(self.num_params, self.hidden_dim, self.out_dim))
        self.bias = Parameter(torch.FloatTensor(self.num_params, 1, self.out_dim))

    def forward(self, x, fact, inp, msg_to, order, fact_type):
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

        if self.fact_type == "B":
            atom_pairs = x[fact.clone()[:, 0]][:, 1:].clone()
            ids = atom_pairs[:, 0] * self.max_num_atoms + atom_pairs[:, 1]
            ids = ids.clone()
        elif self.fact_type == "C":
            msp_nodes = x[fact[:, 0]][:, 2]
            ids = msp_nodes
        elif self.fact_type == "A":
            atom_nodes = x[fact[:, 0]][:, 1]
            ids = atom_nodes

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
    def __init__(self, num_iters=3, in_dim=13, rank=128, num_classes=26, fact_type=None):
        super(FGNet, self).__init__()
        self.rank = rank
        # self.latent_dim = 64
        self.latent_dim = in_dim
        self.bond_type = 4
        self.max_num_atoms = 13
        self.max_msp_index = 1000
        if fact_type not in ["A", "B", "C"]:
            raise Exception("Must specify factor type correctly")
        self.fact_type = fact_type

        if self.fact_type == "B":
            self.num_params = self.max_num_atoms**2
        elif self.fact_type == "C":
            self.num_params = self.max_msp_index
        elif self.fact_type == "A":
            self.num_params = self.max_num_atoms

        self.highorder_func = nn.ModuleList()
        self.max_order = 14
        for i in range(self.max_order):
            self.highorder_func.append(HighOrderNet(
                in_dim=self.latent_dim, out_dim=self.latent_dim,
                hidden_dim=self.rank, fact_type=fact_type,
            ))

        self.params = Parameter(torch.FloatTensor(self.num_params, self.latent_dim, self.rank))
        self.bias = Parameter(torch.FloatTensor(self.num_params, 1, self.rank))

    def forward(self, x, nodes, fact, fact_type=None):
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
            elif self.fact_type == "C":
                msp_nodes = x[fact[:, 0]][:, 2]
                ids = msp_nodes
            elif self.fact_type == "A":
                atom_nodes = x[fact[:, 0]][:, 1]
                ids = atom_nodes

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
            current_msg_x = self.highorder_func[i](x, fact, inp=inp_x,
                                                   msg_to=i, order=sub_fact.size(1), fact_type=fact_type)
            if i == 0:
                node_msg = (nodes_to_edges[i]) @ current_msg_x
            else:
                node_msg += (nodes_to_edges[i]) @ current_msg_x

        return node_msg
