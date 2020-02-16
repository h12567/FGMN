from Models import Encoder, EdgeClassify, EncoderEdgeClassify
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

max_atoms = 13
msp_len = 800
src_vocab = max(1000, max_atoms)
d_model = 512 # size of atom embedding after encorder
heads = 8 # number of heads in multi-head attention
N = 3 # number of loops of encoderLayer
dropout = 0.1
batch_size = 8
num_bonds_prediction = 4 # (no_bond, single, double, triple)

model = EncoderEdgeClassify(src_vocab, d_model, N, heads, dropout, msp_len, max_atoms, num_bonds_prediction)
vertex_arr = np.load("../nist_db_helpers/vertex_arr.npy", allow_pickle=True)
mol_adj_arr = np.load("../nist_db_helpers/mol_adj_arr.npy", allow_pickle=True)
msp_arr = np.load("../nist_db_helpers/msp_arr.npy", allow_pickle=True)

def train_model(epochs):
    model.train()

    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        idxes = np.random.choice(range(len(msp_arr)), batch_size)
        src = torch.from_numpy(msp_arr[idxes].astype(int))
        labels = torch.from_numpy(mol_adj_arr[idxes])
        labels = labels.permute(0, 2, 3, 1)
        src_mask = None
        preds = model(src, src_mask, max_atoms)
        # loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

train_model(100)
