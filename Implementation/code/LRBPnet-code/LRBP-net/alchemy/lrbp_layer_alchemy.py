import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import functional as F
import numpy as np
import scipy.sparse as sp
from torch_scatter import scatter_mean,scatter_add


class HighorderNet(nn.Module):
    def __init__(self, order=3, in_dim=512, out_dim=512, hidden_dim=None, mode='BT'):
        super(HighorderNet, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = self.in_dim if hidden_dim is None else hidden_dim
        self.out_dim = out_dim
        self.order = order
        self.mods = nn.ModuleList()
        for i in range(self.order):
            self.mods.append(nn.Linear(self.in_dim, self.hidden_dim, bias=True))
        self.fc = nn.Linear(self.hidden_dim, out_dim, bias=True)
        self.linear_list = nn.ModuleList()
        
        self.atoms = 7
        self.mode = mode
        assert mode in ('CAT', 'BT', 'CABT', 'CABTA')
        if self.mode == 'CAT':
            self.num_params = self.atoms + 1
        elif self.mode == 'BT':
            self.num_params = (self.order + 1)
        elif self.mode == 'CABT':
            self.num_params = self.atoms * (self.order + 1)
        elif self.mode == 'CABTA':
            self.num_params = self.atoms * self.atoms * (self.order + 1)
        
        self.params = Parameter(torch.FloatTensor(self.num_params, self.hidden_dim, self.out_dim))
        self.bias = Parameter(torch.FloatTensor(self.num_params, 1, self.out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.params.size(2))
        self.params.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inp, msg_to, order, edge_types, atoms, numer, atom_edges):
        isfirst = True
        for i in range(order):
            if i == msg_to:
                continue
            if isfirst:
                fact = ((inp[i]).unsqueeze(-1))
                isfirst = False
            else:
                fact = torch.cat((fact, ((inp[i])).unsqueeze(-1)), -1)      
        fact_prod = fact.prod(-1)
        
        if self.mode == 'CAT':
            ids = atoms
        elif self.mode == 'BT':
            ids = edge_types[:,msg_to]
        elif self.mode == 'CABT':
            ids = (atoms*self.atoms) + edge_types[:,msg_to]
        elif self.mode == 'CABTA':
            ids = (atoms*self.atoms*self.atoms)+ (atom_edges[:,msg_to]*self.atoms) + edge_types[:,msg_to]
          
        weights = self.params.view(self.num_params, self.hidden_dim * self.out_dim)[ids].contiguous()
        bias = self.bias[ids]
        params_cat = weights.view(-1,self.hidden_dim, self.out_dim) #torch.cat(params, dim=0)    
        
        transform_lst = []
        transform = (fact_prod.unsqueeze(1) @ params_cat) + bias
        return transform.squeeze(1)
            

class BPNet(nn.Module):
    def __init__(self, num_iters=3, order=3, in_dim=13, rank=128, num_classes=26, mode='BT'):
        super(BPNet, self).__init__()
        self.latent_dim = in_dim
        self.rank = rank
        self.num_iters = num_iters
        self.num_classes = num_classes
        self.order = order
        self.highorder_func = nn.ModuleList()
        for i in range(self.order):
            self.highorder_func.append(HighorderNet(order=self.order,
                                        in_dim=self.latent_dim, out_dim=self.latent_dim,
                                        hidden_dim=self.rank, mode=mode))
        self.linear_list = nn.ModuleList()
        self.atoms = 7
        self.mode = mode
        assert mode in ('CAT', 'BT', 'CABT', 'CABTA')
        if self.mode == 'CAT':
            self.num_params = self.atoms + 1
        elif self.mode == 'BT':
            self.num_params = (self.order + 1)
        elif self.mode == 'CABT':
            self.num_params = self.atoms * (self.order + 1)
        elif self.mode == 'CABTA':
            self.num_params = self.atoms * self.atoms * (self.order + 1)
        
        self.params = Parameter(torch.FloatTensor(self.num_params, self.latent_dim, self.rank))
        self.bias = Parameter(torch.FloatTensor(self.num_params, 1, self.rank))
        self.reset_parameters()

    def reset_parameters(self):
        for (name, module) in self._modules.items():
            if isinstance(module, nn.ModuleList):
                for mod in module:
                    mod.reset_parameters() 
                    
        stdv = 1. / math.sqrt(self.params.size(2))
        self.params.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, nodes, edges, edge_types, atoms, atom_edges):#, atom_types):         
        numer = torch.arange(0, edges.size(0), dtype=torch.long, device=nodes.device)
        if edges.size(0) ==0:
            return nodes

        nodes_to_edges = []
        rnodes_l = []
        params = []
        bias_lst = []
        transform_lst = []
        for i in range(edges.size(1)):
            nodes_to_edges1 = nodes.data.new(nodes.size(0), edges.size(0)).zero_()
            nodes_to_edges1.view(-1)[edges[:, i] * edges.size(0) + numer] = 1
            nodes_to_edges.append(nodes_to_edges1)
            rnodes = nodes[edges[:,i]]
        
            if self.mode == 'CAT':
                ids = atoms
            elif self.mode == 'BT':
                ids = edge_types[:,i]
            elif self.mode == 'CABT':
                ids = (atoms*self.atoms) + edge_types[:,i]
            elif self.mode == 'CABTA':
                ids = (atoms*self.atoms*self.atoms)+ (atom_edges[:,i]*self.atoms) + edge_types[:,i]
                
            weights = self.params.view(self.num_params,self.latent_dim * self.rank)[ids].contiguous()
            params = weights.view(-1,self.latent_dim, self.rank)
            bias = self.bias[ids]
            
            transform = (rnodes.unsqueeze(1) @ params) + bias
            transform_lst.append(transform.squeeze(1).unsqueeze(0)) #r x rk
        
        transform_cat = F.relu(torch.cat(transform_lst, dim=0)) #ord x r x rk
        
        for itr in range(self.num_iters):
            inp_x = transform_cat
            msg_x = []
            for i in range(edges.size(1)):
                msg_x.append(self.highorder_func[i](inp=inp_x, msg_to=i, order=edges.size(1),
                    edge_types=edge_types, atoms=atoms,numer=numer,atom_edges=atom_edges))
                if i == 0:
                    node_msg = (nodes_to_edges[i]) @ msg_x[i]
                else:
                    node_msg = node_msg + (nodes_to_edges[i] ) @ msg_x[i]
        
        return node_msg

