import torch
import torch.nn as nn
from tsfm.Layers import EncoderLayer
from tsfm.Layers import DecoderLayer
from tsfm.Embed import Embedder, PositionalEncoder
from tsfm.Sublayers import Norm
import copy
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import tsfm.getInput as getInput

#Hyperparameters
learning_rate = 0.01
gamma = 0.99
action_num = 3
max_atoms = 13


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout,enc_padding_idx):
        super().__init__()
        self.N = N
        #We do embedding part in getInput.py and no need for positional here
        #self.embed_atom = Embedder(2, d_model)
        #self.embed_msp = Embedder(vocab_size, d_model)
        #self.pe = PositionalEncoder(d_model=d_model, max_seq_len=13,dropout=dropout)
        self.embedding = nn.Embedding(vocab_size, d_model//2, padding_idx=enc_padding_idx)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x=self.embedding(src).view(src.size(0),src.size(1),-1) #[N, 30, d_model]
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        #We do embedding part in getInput.py and no need for positional here
        #self.embed_atom = Embedder(2, d_model)
        #self.embed_msp = Embedder(vocab_size, d_model)
        #self.pe = PositionalEncoder(d_model=d_model, max_seq_len=13,dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, dec_input,enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        for i in range(self.N):
            x = self.layers[i](dec_input, enc_output,slf_attn_mask,dec_enc_attn_mask)
        return x

class EdgeClassify(nn.Module): #orignial
    def __init__(self, msp_len, max_atoms, d_model, num_bonds_prediction, k):
        super().__init__()
        self.ll1 = nn.Linear(d_model, max_atoms)
        self.ll2 = nn.Linear(max_atoms + k, max_atoms)
        self.fl = nn.Linear(max_atoms * 2, num_bonds_prediction)

    def forward(self, e_output, max_atoms):
        reduce_e = self.ll1(e_output)  # [bs, 43,13]
        reduce_e = reduce_e.permute(0, 2, 1)  # [N,13,43]
        #reduce_e = self.ll2(reduce_e)  # [bs, 13,13]
        reduce_e = reduce_e[:,:,:max_atoms] # [bs, 13,13]
        reduce_e = reduce_e.unsqueeze(2)  # Shape: (batch, max_atoms, 1, e_out_dim) [bt,13,1,d_m]
        repeat_e = reduce_e.repeat(1, 1, max_atoms, 1)  # (batch, max_atoms, max_atoms, e_out_dim) [bt,13,13,d_m]
        final_e = torch.zeros(
            (repeat_e.shape[0], repeat_e.shape[1], repeat_e.shape[2], 4),dtype=torch.float32)  # [bt,13,13,d_m*2]
        for i in range(max_atoms):
            for j in range(max_atoms):
                if i < j:
                    final_e[:, i, j, :] = self.fl(torch.cat((repeat_e[:, i, i, :], repeat_e[:, j, j, :]), dim=1))
                elif i>j:
                    final_e[:, i, j, :] = self.fl(torch.cat((repeat_e[:, j, j, :], repeat_e[:, i, i, :]), dim=1))
        #final_e = self.fl(final_e)  # [8,13,13,4]
        return final_e

class EdgeClassify2(nn.Module): #with d_model
    def __init__(self, msp_len, max_atoms, d_model, num_bonds_prediction,k):
        super().__init__()
        self.ll1 = nn.Linear(max_atoms+k, max_atoms)
        self.fl = nn.Linear(d_model * 2, num_bonds_prediction)

    def forward(self, e_output, max_atoms):
        x = e_output
        x = x.permute(0, 2, 1)  # [N,dim,43]
        x = self.ll1(x) #[N, dim, 13]
        x = x.permute(0, 2, 1)  # [N,13,dim]]
        x = x.unsqueeze(2) #(batch, max_atoms, 1, e_out_dim) [bt,13,1,d_m]
        x = x.repeat(1, 1, max_atoms, 1)  # (batch, max_atoms, max_atoms, e_out_dim) [bt,13,13,d_m]
        final_e = torch.tensor((), dtype=torch.float32)
        final_e = final_e.new_ones(
            (x.shape[0], x.shape[1], x.shape[2], 2 * x.shape[3]))  # [bt,13,13,d_m*2]
        for i in range(max_atoms):
            for j in range(max_atoms):
                if i < j:
                    final_e[:, i, j, :] = torch.cat((x[:, i, i, :], x[:, j, j, :]), dim=1)
                else:
                    final_e[:, i, j, :] = torch.cat((x[:, j, j, :], x[:, i, i, :]), dim=1)
        final_e = self.fl(final_e)  # [8,13,13,4]
        return final_e

class EdgeDecoderClassify(nn.Module): #with decoder
    def __init__(self, msp_len, max_atoms, d_model, num_bonds_prediction, k):
        super().__init__()
        self.ll1 = nn.Linear(d_model, max_atoms)
        self.ll2 = nn.Linear(k, max_atoms)
        self.fl = nn.Linear(max_atoms * 2, num_bonds_prediction)

    def forward(self, x, max_atoms):
        reduce_e = self.ll1(x)  # [bs, k,13]
        reduce_e = reduce_e.permute(0, 2, 1)  # [N,13,43]
        reduce_e = self.ll2(reduce_e)  # [bs, 13,13]
        reduce_e = reduce_e.unsqueeze(2)  # Shape: (batch, max_atoms, 1, e_out_dim) [bt,13,1,d_m]
        repeat_e = reduce_e.repeat(1, 1, max_atoms, 1)  # (batch, max_atoms, max_atoms, e_out_dim) [bt,13,13,d_m]
        final_e = torch.tensor((), dtype=torch.float32)
        final_e = final_e.new_ones(
            (repeat_e.shape[0], repeat_e.shape[1], repeat_e.shape[2], 2 * repeat_e.shape[3]))  # [bt,13,13,d_m*2]
        for i in range(max_atoms):
            for j in range(max_atoms):
                if i < j:
                    final_e[:, i, j, :] = torch.cat((repeat_e[:, i, i, :], repeat_e[:, j, j, :]), dim=1)
                else:
                    final_e[:, i, j, :] = torch.cat((repeat_e[:, j, j, :], repeat_e[:, i, i, :]), dim=1)
        final_e = self.fl(final_e)  # [8,13,13,4]
        return final_e

class EdgeCNNClassify(nn.Module):
    def __init__(self, msp_len, max_atoms, d_model, num_bonds_prediction, k):
        super().__init__()
        self.ll1 = nn.Linear(d_model,100)
        self.ll2 = nn.Linear(max_atoms + k, max_atoms)
        self.fl = nn.Linear(4, num_bonds_prediction)
        self.conv = nn.Conv2d(1,4,kernel_size=(1,100*2),stride=(1,100*2),padding=0)
        self.relu = nn.ReLU()

    def forward(self, x, max_atoms):
        length = 100
        reduce_e = x[:,:13,:] #[bs, 13, d_m]
        reduce_e = self.ll1(reduce_e)  # [bs, 13,100]
        #reduce_e = reduce_e.permute(0, 2, 1)  # [N,13,13]
        #reduce_e = self.ll2(reduce_e)  # [bs, 13,13]
        reduce_e = reduce_e.unsqueeze(2)  # Shape: (batch, max_atoms, 1, e_out_dim) [bt,13,1,d_m]
        repeat_e = reduce_e.repeat(1, 1, max_atoms, 1)  # (batch, max_atoms, max_atoms, e_out_dim) [bt,13,13,d_m]
        cnn_input = torch.zeros((repeat_e.shape[0],1,max_atoms,max_atoms*length*2), dtype=torch.float32)
        for i in range(max_atoms):
            for j in range(max_atoms):
                if i < j:
                    cnn_input[:, 0,i, j*2*length:(j+1)*2*length] = torch.cat((repeat_e[:, i, i, :], repeat_e[:, j, j, :]), dim=1)
                else:
                    cnn_input[:, 0, i, j * 2*length:(j + 1) * 2*length] = torch.cat((repeat_e[:, j, j, :], repeat_e[:, i, i, :]),
                                                                        dim=1)
        final_e = self.conv(cnn_input)
        final_e = final_e.permute(0,2,3,1) #[batch,13,13,4]
        #final_e = self.fl(final_e)
        return final_e

class EncoderEdgeClassify(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, msp_len, max_atoms, num_bonds_prediction,k):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, N, heads, dropout)
        self.edge_classify = EdgeClassify(msp_len, max_atoms, d_model, num_bonds_prediction,k)

    def forward(self, src, mask, max_atoms):
        output = self.encoder(src, mask) #[batch, 43, 18]
        output = self.edge_classify(output, max_atoms)
        return output

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv1d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv1d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
class CNN(nn.Module):
    def __init__(self, input_channel,output_channel):
        super(CNN, self).__init__()
        self.entry = nn.Sequential(
            depthwise_separable_conv(input_channel, output_channel, kernel_size=5, padding=4),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(True)
        )
        self.entry_flow_residual = nn.Conv1d(output_channel, output_channel, kernel_size=1, stride=1, padding=0)

        self.entry2 = nn.Sequential(
            depthwise_separable_conv(input_channel, output_channel, kernel_size=3, padding=2),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(True)
        )
        self.entry2_flow_residual = nn.Conv1d(output_channel, output_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        entry1_output = self.entry(x)*self.entry_flow_residual(x)
        entry2_output = self.entry2(x) * self.entry2_flow_residual(x)
        output = torch.cat((entry1_output,entry2_output),dim=1)
        return output


class CNNRNNModel(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, msp_len, max_atoms, num_bonds_prediction,k):
        super().__init__()
        self.encoder1 = CNN(5,32) #[batch, 32,]
        self.edge_classify = EdgeClassify2(msp_len, max_atoms, d_model, num_bonds_prediction,k)

    def forward(self, src, mask, max_atoms):
        output = self.encoder(src) #[batch, 5, 18]
        output = self.edge_classify(output, max_atoms)
        return output


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, msp_len, max_atoms, num_bonds_prediction,k):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, N, heads, dropout)
        self.decoder = Decoder(vocab_size, d_model, N, heads, dropout)
        self.edge_classify = EdgeDecoderClassify(msp_len, max_atoms, d_model, num_bonds_prediction,k)

    def forward(self, src, dec_input,mask, max_atoms):
        enc_output = self.encoder(src,mask) #[batch, 5, 18]
        output = self.decoder(enc_output,dec_input)
        output = self.edge_classify(output, max_atoms)
        return output

class ImportanceEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        #We do embedding part in getInput.py and no need for positional here
        #self.embed_atom = Embedder(2, d_model)
        #self.embed_msp = Embedder(vocab_size, d_model)
        #self.pe = PositionalEncoder(d_model=d_model, max_seq_len=13,dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.last = EncoderLayer(d_model, heads=1, dropout=0.1)
        self.norm = Norm(d_model)
        self.ff = nn.Linear(608*2, 4)

    def forward(self, src, mask):
        x=src #[N, 13, 47] float32
        for i in range(self.N):
            x,__ = self.layers[i](x, mask)
        __,scores= self.last(x,mask) #[batch, head=1, 43,43]
        x = x[:,:13,:]
        importance = scores.view(-1,43,43)[:,:13,:13] #[batch, 13, 13]
        _,index = torch.topk(importance,k=4,dim=2)
        out = torch.tensor(([[[[1,0,0,0]]*13]*13]*8),dtype=torch.float32)
        for b in range(src.size(0)):
            for i in range(13):
                for j in range(4):
                    id = int(index[b, i, j])
                    temp = torch.cat((x[b, i], x[b, id]))  # [608*2]
                    out[b, i, id] = self.ff(temp)  # [batch, 4]
        return out

class GraphDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout,dec_padding_idx):
        super().__init__()
        self.N = N
        #We do embedding part in getInput.py and no need for positional here
        self.embedding = nn.Embedding(vocab_size, 5, padding_idx=dec_padding_idx)
        self.pe = PositionalEncoder(d_model=d_model, max_seq_len=13+78,dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, dec_input,enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_input = self.embedding(dec_input)
        dec_input = dec_input.view(dec_input.size(0),dec_input.size(1),-1)
        dec_input = self.pe(dec_input) #[8, 78+13=91, 128]
        for i in range(self.N):
            x = self.layers[i](enc_output,dec_input, slf_attn_mask,dec_enc_attn_mask)
        return x



# class Classify(nn.Module):
#     def __init__(self, enc_vocab_size, dec_vocab_size,d_model, N, heads, dropout, enc_padding_idx,dec_padding_idx):
#         super().__init__()
#         self.encoding_padding_idx = enc_padding_idx
#         self.decoding_padding_idx = dec_padding_idx
#         self.encoder = Encoder(enc_vocab_size,d_model,N,heads,dropout,enc_padding_idx)
#         self.decoder = GraphDecoder(dec_vocab_size,d_model,N,heads,dropout,dec_padding_idx)
#         self.ll =  nn.Linear(d_model, 3)
#
#     def forward(self,dec_input, src=None,enc_out=None):
#         if enc_out is None:
#             self.src_mask=None#get_pad_mask(src[:,:,1],self.encoding_padding_idx)
#             enc_out = self.encoder(src,self.src_mask) #torch.Size([8, 30, 120])
#         self.dec_mask = None#get_pad_mask(dec_input[:,:,1],self.decoding_padding_idx)
#         dec_out = self.decoder(dec_input, enc_out,slf_attn_mask=self.dec_mask, dec_enc_attn_mask=self.src_mask)
#         probs = torch.sigmoid(self.ll(dec_out)) #[8,78,3]
#         return probs,enc_out

class Policy(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout,dec_padding_idx,dec_len):
        super(Policy, self).__init__()
        self.N = N
        self.embedding = nn.Embedding(vocab_size, d_model//dec_len, padding_idx=dec_padding_idx)
        self.pe = PositionalEncoder(d_model=d_model, max_seq_len=(max_atoms**2+max_atoms)//2, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
        self.ll = nn.Linear(d_model, 3)


    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_input = self.embedding(dec_input)
        dec_input = dec_input.view(dec_input.size(0), dec_input.size(1), -1)
        dec_input = self.pe(dec_input)  # [8, 78+13=91, 128]
        for i in range(self.N):
            x = self.layers[i](enc_output, dec_input, slf_attn_mask, dec_enc_attn_mask)
        output = self.ll(x).view(dec_input.size(0),-1)
        output = F.softmax(output, dim=1)
        return output


class Classify(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size,d_model, N, heads, dropout, enc_padding_idx,dec_padding_idx,dec_len):
        super().__init__()
        self.encoding_padding_idx = None#enc_padding_idx
        self.decoding_padding_idx = None#dec_padding_idx
        self.encoder = Encoder(enc_vocab_size,d_model,N,heads,dropout,enc_padding_idx)
        self.decoder = Policy(dec_vocab_size,d_model,N,heads,dropout,dec_padding_idx,dec_len)
        # Episode policy and reward history
        self.policy_history = []
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        self.gamma = gamma


    def forward(self,dec_input, src=None,enc_out=None,actions=None,atoms = None):
        if enc_out is None:
            self.src_mask=None#get_pad_mask(src[:,:,1],self.encoding_padding_idx)
            enc_out = self.encoder(src,self.src_mask) #torch.Size([8, 30, 120])
        self.dec_mask = None#get_pad_mask(dec_input[:,:,1],self.decoding_padding_idx)
        probs = self.decoder(dec_input, enc_out,slf_attn_mask=self.dec_mask, dec_enc_attn_mask=self.src_mask)
        return probs,enc_out

