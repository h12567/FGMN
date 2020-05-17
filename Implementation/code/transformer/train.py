from tsfm.Models import Encoder, EdgeClassify, EncoderEdgeClassify,TransformerModel,Classify
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import tsfm.getInput as getInput
from tsfm.getInput import GetInput,getBondNum, GetDecoderInput, GetMSInput,GetEdge,find
from torch.autograd import Variable

max_atoms = 13
atom_type=6
msp_len = 800
src_vocab = 1000
d_model = 180 # size of atom embedding after encorder d_model%heads=0
heads =  4# number of heads in multi-head attention
N = 4 # number of loops of encoderLayer
dropout = 0.1
batch_size = 8
num_bonds_prediction = 4 # (no_bond, single, double, triple)
k=30
enc_padding_idx = 0
dec_padding_idx = 0
enc_vocab_size, dec_vocab_size = 1000, 30

vertex_arr = np.load("../tsfm/vertex_arr_sort_svd.npy", allow_pickle=True) #225
mol_adj_arr = np.load("../tsfm/mol_adj_arr_sort_svd.npy", allow_pickle=True)
msp_arr = np.load("../tsfm/msp_arr_sort.npy", allow_pickle=True)
msp_all_data = getInput.GetMSInput(msp_arr,k) #Tensor
mol_adj_data, dec_len = getInput.GetDecoderEdges(np.zeros((len(mol_adj_arr),max_atoms,max_atoms)),vertex_arr,max_atoms,padding_idx=dec_padding_idx,type="input") #Tensor
labels_all_data,_ = getInput.GetDecoderEdges(mol_adj_arr,vertex_arr,max_atoms,padding_idx=dec_padding_idx,type="output") #Tensor

model = Classify(enc_vocab_size, dec_vocab_size,d_model, N, heads, dropout, enc_padding_idx,dec_padding_idx,dec_len)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
new_inputs = None
criterion = nn.BCELoss()

def update_policy(policy,seq_len,atom_input,labels,edge_lists):
    pred_mat = getInput.edge_mat(seq_len,edge_lists,atom_input)
    policy_gradient = []
    losses = []
    for i in range(seq_len):
        R = 0
        rewards = []
        # Discount future rewards back to the present using gamma
        for r in policy.reward_episode[:][i][::-1]:
            R = r + policy.gamma * R
            rewards.insert(0, R)
        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        # Calculate loss
        log_probs = policy.policy_history[:][i]
        idx = 0
        for log_prob, Gt in zip(log_probs, rewards):
            policy_gradient.append(-log_prob * Gt)
            idx += 1
            if idx == len(atom_input[i]):break
    # Update network weights
    optimizer.zero_grad()
    loss = torch.stack(policy_gradient).sum()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = []
    policy.reward_episode = []

def accuracy_of_graphs(preds,labels,a_num,batch_size): #[13,13]
    acc=0
    for i in range(batch_size):
        g1 = preds[:, i]
        g2 = labels[i,1]
        count = 0
        for m in range(max_atoms):
            for n in range(max_atoms):
                if g1[m, n] == g2[m, n]: count += 1
        acc += count / (max_atoms * max_atoms)
    return acc


def accuracys(epoch,preds,labels,batch_size,num_atoms): #[13,13]
    pred_mat = np.zeros((batch_size, 13, 13))
    preds = torch.Tensor(preds)
    acc = []
    for i in range(batch_size):
        g1 = preds[:,i] #[edges per mol] [89,55,77,...] 13
        for j in range(num_atoms[i]):
            if g1[j][0] == -1: continue
            x,y = getInput.find(int(g1[j][0]))
            pred_mat[i,x,y] = int(g1[j][1])
            pred_mat[i, y, x] = int(g1[j][1])
        count = 0
        tot = 0
        for m in range(num_atoms[i]-1):
            for n in range(m+1,num_atoms[i]):
                if labels[i][m][n] != 0:
                    tot += 1
                    if labels[i][m][n] == pred_mat[i][m][n]: count += 1
        acc.append(count / tot)
    if epoch % 1 == 0:
        print("epoch:{:1d}".format(epoch))
        print(labels[0]) #label
        print(pred_mat[0]) #pred
        print("acc: ",acc)
    elif epoch % 1 == 0:
        print("acc: ",acc)
    return acc

def train_model(model,epoch,num):
    model.train()
    # src_data = new_inputs[:num]
    # #src_data = new_inputs[:num, :13, :]
    # label_data = mol_adj_arr[:num]
    # #dec_input_data = new_inputs[:num,13:,:]
    # atom_data = atom_num[:num]
    total_loss=0
    total_acc=0
    msp_data = msp_all_data[:num]
    atom_data = vertex_arr[:num]
    edge_data = mol_adj_data[:num]
    labels_data = labels_all_data[:num]

    for batch, i in enumerate(range(0, len(msp_data), batch_size)):
        seq_len = min(batch_size, len(msp_data)- i)
        msp_input = msp_data[i:i+seq_len]
        atom_input = atom_data[i:i+seq_len]
        edge_input = edge_data[i:i+seq_len]#[[0 0 1],[..]]
        labels = labels_data[i:i+seq_len].clone()
        #a_num = atom_data[i:i+seq_len]
        #dec_input = torch.from_numpy(dec_input_data[i:i+seq_len])
        #labels = torch.from_numpy(label_data[i:i+seq_len]).long() #[batch, max-atom,max-atom]=[8,13,13]
        dec_input = edge_input#GetDecoderInput(edge_input,atom_input,d_model) # 72 edges + <=13 atoms Tensor
        edge_list = np.zeros((max_atoms,seq_len),dtype=np.int32)
        #preds = model(src, src_mask, max_atoms) #[8,13,13,4] batch-size, outwidth, outheight, out-channel
        #preds = model(src,dec_input, src_mask, max_atoms)  # [8,13,13,4] batch-size, outwidth, outheight, out-channel
        for j in range(max_atoms):
            optimizer.zero_grad()
            #print(j)
            if j ==0:
                probs, enc_output = model(dec_input,src=msp_input, enc_out=None)
            else:
                probs, enc_output = model(dec_input,src=None, enc_out=enc_output)
            t_labels = labels[:,:,1:4]
            loss = criterion(probs.contiguous().view(-1), t_labels.contiguous().view(-1))
            print(loss)
            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()
            probs = probs.contiguous().view(probs.size(0),-1)
            actions = []
            for a in range(seq_len):
                while(1):
                    eps = 1/(j+1)
                    sample = random.random()
                    prob = F.softmax(probs[a])
                    if sample>eps:
                        action = int(torch.argmax(prob))
                    else:
                        m = torch.distributions.Categorical(prob)
                        action = int(m.sample())  # 8
                    x, y = find(action // 3)
                    # actions.append(action)
                    # break
                    if x<len(atom_input[a]) and y <len(atom_input[a]) and (action // 3) not in []:
                        edge_list[j,a]=action
                        actions.append(action)
                        break
            for a in range(len(actions)):
                labels[a, actions[a] // 3] = 0
            dec_input = getInput.GetDecoderEdges(np.zeros((seq_len,max_atoms,max_atoms)),
                                                 vertex_arr,max_atoms,padding_idx=dec_padding_idx,type="input",actions=actions)
            #dec_input = GetDecoderInput(edge_input,atom_input,d_model,actions)
        #loss = criterion(preds.contiguous().view(-1,num_bonds_prediction), labels.view(-1))
        #loss.backward()
        #optimizer.step()
        #total_loss+=loss.item()
        if batch==0:
            acc = accuracys(epoch,edge_list,labels_data[i:i+seq_len],batch_size=seq_len)
        #total_acc+=acc
    # if epoch % 10 == 0:
    #     print("avg_acc: ",total_acc/num)
    #     print("total_loss:",total_loss/num)

def evaluate(model,test_idx):
    print("eval!")
    model.eval()
    src_data = new_inputs[test_idx]
    label_data = mol_adj_arr[test_idx]
    with torch.no_grad():
        for i in range(0, len(src_data), batch_size):
            print(i)
            seq_len = min(batch_size, len(src_data) - i)
            src = torch.from_numpy(src_data[i:i + seq_len].astype(int))
            labels = torch.from_numpy(label_data[i:i + seq_len]).long()  # [batch, max-atom,max-atom]=[8,13,13]
            src_mask = None
            preds = model(src, src_mask, max_atoms)
            loss = criterion(preds.view(-1, num_bonds_prediction), labels.view(-1))
            print("loss: ", loss.item())

def test(model,test_idx):
    print("Test!")
    model.eval()
    src_data = new_inputs[test_idx]
    label_data = mol_adj_arr[test_idx]
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for i in range(0, len(src_data), batch_size):
            #print(i)
            seq_len = min(batch_size, len(src_data) - i)
            src = torch.from_numpy(src_data[i:i+seq_len])
            labels = torch.from_numpy(label_data[i:i + seq_len]).long()  # [batch, max-atom,max-atom]=[8,13,13]
            src_mask = None
            preds = model(src, src_mask, max_atoms)
            acc = accuracy_of_graphs(preds, labels, batch_size=seq_len)
            print(labels[0])
            print(torch.argmax(preds[0],dim=2))
            loss = criterion(preds.view(-1, num_bonds_prediction), labels.view(-1))
            total_acc+=acc
            total_loss+=loss.item()
        print("avg_acc: ", total_acc/len(test_idx))
        print("total_loss: ", total_loss)

def train_pg(model,epoch,num):
    model.train()
    msp_data = msp_all_data[:num]
    atom_data = vertex_arr[:num]
    edge_data = mol_adj_data[:num]

    for batch, i in enumerate(range(0, len(msp_data), batch_size)):
        seq_len = min(batch_size, len(msp_data)- i)
        msp_input = msp_data[i:i+seq_len]
        atom_input = atom_data[i:i+seq_len]
        edge_input = edge_data[i:i+seq_len]#[[0 0 1],[..]]
        labels = mol_adj_arr[i:i+seq_len]
        dec_input = edge_input
        edge_lists = [[[0,2]]*seq_len,[[1,1]]*seq_len] #[position, type]
        for j in range(max_atoms-2):
            print("j: ",j)
            if j ==0:
                probs, enc_output = model(dec_input,src=msp_input, enc_out=None)
            else:
                probs, enc_output = model(dec_input,src=None, enc_out=enc_output) # [batch, 78*3=234]
            log_probs, rewards, actions, edge_list = getInput.select_action(probs,labels,edge_lists,atom_input,seq_len,max_atoms)
            dec_input,_ = getInput.GetDecoderEdges(np.zeros((seq_len,max_atoms,max_atoms)),
                                                 vertex_arr,max_atoms,padding_idx=dec_padding_idx,type="input",actions=actions)
            # Add log probability of our chosen action to our history
            model.policy_history.append(log_probs)
            model.reward_episode.append(rewards)
            edge_lists.append(edge_list)
        num_atoms = [len(i) for i in atom_input]
        #update_policy(model,seq_len,atom_input,labels,edge_lists)
        min_loss = getInput.select_loss(model,seq_len,atom_input,labels,edge_lists)
        optimizer.zero_grad()
        min_loss.backward()
        optimizer.step()
        if batch==0:
            print("loss:",model.loss_history)
            acc = accuracys(epoch,edge_lists,labels,seq_len,num_atoms)


def transformer(epoch,num):
    for i in range(1,1+epoch):
        print("epoch:",i)
        train_pg(model,i,num)
        #evaluate(model, range(120, 121))
    torch.save(model.state_dict(),'model5.pkl')
    #model.load_state_dict(torch.load('model5.pkl'))
    test(model,range(160, 210))


transformer(400,num=16)

