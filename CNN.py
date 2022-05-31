#!/usr/bin/env python3.7
#coding=utf-8

"""
Author: ML Groups
Created Date:  2022/01/06
Last Modified: 2022/01/10
"""

import numpy as np
import torch.utils.data as Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam


#
class MLP(nn.Module):  
     """ class for final assignment 

    Properties:
        a MLP model with 5 fully connected layerof 32 interconnected nodes
        Activation function using ReLU
        Input: tensor([time, init_x2_crd_x, init_x2_crd_y])
        Output: tensor([t_x1_crd_x, t_x1_crd_y, t_x2_crd_x, t_x2_crd_y])
    """
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(3,32)
        self.hidden2 = nn.Linear(32,32)
        self.hidden3 = nn.Linear(32,32)
        self.hidden4 = nn.Linear(32,32)
        self.hidden5 = nn.Linear(32,32)
        self.hidden6 = nn.Linear(32,32)
        self.hidden7 = nn.Linear(32,4)
    def forward(self,input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = F.relu(out)
        out = self.hidden3(out)
        out = F.relu(out)
        out = self.hidden4(out)
        out = F.relu(out)
        out = self.hidden5(out)
        out = F.relu(out)
        out = self.hidden6(out)
        out = F.relu(out)
        out = self.hidden7(out)
        return out


# Initiate MLP model
net = MLP().double()


# Define fit function, using MAE as loss fuction
def fit(trainloader, valloader, lr, max_epoch = 10):
    epoch_losses = []
    
    loss_fn = nn.L1Loss()
    optimizer = Adam(net.parameters(), lr = lr) # Using Adam optimizer
    batches_per_epoch = len(trainloader) # Loss function, adjustable
    
    for epoch in range(max_epoch):
        epoch_loss = 0
        for x,patch in enumerate(trainloader):
            batch_loss=0
            for i in range(401):
                optimizer.zero_grad()
                pred = net(patch[0][i])
                loss = loss_fn(pred, patch[1][i])
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            print("Epoch %d, Batch %d loss: %f"%(epoch, x, batch_loss/401))
            epoch_loss+=batch_loss
        epoch_losses.append(epoch_loss)
        print("\n##### Epoch %d average loss: "%epoch, epoch_loss/batches_per_epoch, ' #####\n')


# parms
SEED = 20220106
train_size = 401*75*used_trajs
valid_size = 401*25*used_trajs
lr = 0.0001
batch_size = 401  # 1 traj as 1 batch




#read trajactory file and convert to dataset
# Using 80~100 traj file as trainset & validset
used_trajs = 20
X= np.zeros((100*401*used_trajs,3))
Y= np.zeros((100*401*used_trajs,4))
x2_init = np.zeros((2))
for file in range(used_trajs):
    f1 = open(f'./data/dNB.{100-file}.traj', 'r')
    for i in range(401*100):
        line1 = [float(i) for i in f1.readline().split()]
        line2 = [float(i) for i in f1.readline().split()]
        f1.readline()
        f1.readline()
        if (i%401 ==0):
            x2_init = line2[4:6]
        X[i+file*401*100] = line1[2:3] + x2_init
        Y[i+file*401*100] = line1[4:6] + line2[4:6]
        
torch_dataset = Data.TensorDataset(torch.from_numpy(X),torch.from_numpy(Y))
trainset, valset = Data.random_split(torch_dataset, [train_size, valid_size], generator=torch.Generator().manual_seed(SEED))
trainloader = Data.DataLoader(trainset, batch_size, shuffle = False, drop_last = False)
valloader = Data.DataLoader(valset, batch_size, shuffle = False, drop_last = False)
print("dataset size: ", len(torch_dataset))
print("Trainset size: ", len(trainset))
print("Validation set size: ", len(valset))

# fit model
fit(trainloader, valloader, lr = lr,max_epoch = 10)

# save model

import os
os.makedirs("./checkpoints", exist_ok = True)
# save parameters only
torch.save(net.state_dict(), "./checkpoints/model_ann2.pt")
# load parameters
#net.load_state_dict(torch.load("./checkpoints/model_ann2.pt"))


