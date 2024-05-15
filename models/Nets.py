#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import math

# Batch_Size
hidden_sizes = [128, 500]
# hidden_sizes = [64, 500]
output_size = 10

class ClientModelmnist(nn.Module):
    def __init__(self):
        super(ClientModelmnist, self).__init__()
        self.lin = nn.Linear(392,64)
        
    def forward(self, x):
        x = self.lin(x)
        x = nn.functional.relu(x)
        return x
    
class ServerModelmnist(nn.Module):
    def __init__(self):
        super(ServerModelmnist, self).__init__()
        self.lin2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.lin3 = nn.Linear(hidden_sizes[1], output_size)
        self.sft = nn.LogSoftmax(dim=1) 
        
    def forward(self, x):
        x = self.lin2(x)
        x = nn.ReLU()(x)
        x = self.lin3(x)
        x = self.sft(x)
        return x
    

class VHFLGroup(nn.Module):
    def __init__(self):
        super(VHFLGroup, self).__init__()
        self.client_model_1 = ClientModelmnist()
        self.client_model_2 = ClientModelmnist()
        self.server = ServerModelmnist()

        self.opt_c1 = torch.optim.SGD(params=self.client_model_1.parameters(),lr=0.01)
        self.opt_c2 = torch.optim.SGD(params=self.client_model_2.parameters(),lr=0.01)
        self.opt_s = torch.optim.SGD(params=self.server.parameters(),lr=0.1)

    def forward(self, x1, x2):
        x1 = self.client_model_1(x1)
        x2 = self.client_model_2(x2)
        x = torch.cat((x1, x2), 1)
        x = self.server(x)

        return x
    

    # def train(self):
    #     self.client_model_1.train()
    #     self.client_model_2.train()
    #     self.server.train()


    def zero_grad(self):
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()
        self.opt_s.zero_grad()

    def step_opts(self):
        self.opt_c1.step()
        self.opt_c2.step()
        self.opt_s.step()
    
