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


class ClientModelCifar(nn.Module):
    def __init__(self):
        super(ClientModelCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

class ServerModelCifar(nn.Module):
    def __init__(self, num_classes):
        super(ServerModelCifar, self).__init__()
        self.fc1 = nn.Linear(32 * 1 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.pool = nn.MaxPool2d(2,2)
    def forward(self, x):
        x = x.view(-1, 32 * 1 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




class VHFLGroup(nn.Module):
    def __init__(self, opt, lr):
        super(VHFLGroup, self).__init__()
        self.client_model_1 = ClientModelmnist()
        self.client_model_2 = ClientModelmnist()
        self.server = ServerModelmnist()

        if opt == "SGD":
            self.opt_c1 = torch.optim.SGD(params=self.client_model_1.parameters(),lr=lr)
            self.opt_c2 = torch.optim.SGD(params=self.client_model_2.parameters(),lr=lr)
            self.opt_s = torch.optim.SGD(params=self.server.parameters(),lr=lr)
        elif opt == "Adam":
            self.opt_c1 = torch.optim.Adam(params=self.client_model_1.parameters(),lr=lr)
            self.opt_c2 = torch.optim.Adam(params=self.client_model_2.parameters(),lr=lr)
            self.opt_s = torch.optim.Adam(params=self.server.parameters(),lr=lr)

    def forward(self, x1, x2):
        x1 = self.client_model_1(x1)
        x2 = self.client_model_2(x2)
        x = torch.cat((x1, x2), 1)
        x = self.server(x)

        return x

    def zero_grad(self):
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()
        self.opt_s.zero_grad()

    def step_opts(self):
        self.opt_c1.step()
        self.opt_c2.step()
        self.opt_s.step()

    def to(self, device):
        self.client_model_1.to(device)
        self.client_model_2.to(device)
        self.server.to(device)
    
class VHFLGroupCifar(nn.Module):
    def __init__(self, opt, lr):
        super(VHFLGroupCifar, self).__init__()
        self.client_model_1 = ClientModelCifar()
        self.client_model_2 = ClientModelCifar()
        self.server = ServerModelCifar(10)

        if opt == "SGD":
            self.opt_c1 = torch.optim.SGD(params=self.client_model_1.parameters(),lr=lr)
            self.opt_c2 = torch.optim.SGD(params=self.client_model_2.parameters(),lr=lr)
            self.opt_s = torch.optim.SGD(params=self.server.parameters(),lr=lr)
        elif opt == "Adam":
            self.opt_c1 = torch.optim.Adam(params=self.client_model_1.parameters(),lr=lr)
            self.opt_c2 = torch.optim.Adam(params=self.client_model_2.parameters(),lr=lr)
            self.opt_s = torch.optim.Adam(params=self.server.parameters(),lr=lr)

    def forward(self, x1, x2):
        x1 = self.client_model_1(x1)
        x2 = self.client_model_2(x2)
        x = torch.cat((x1, x2), 1)
        x = self.server(x)

        return x

    def zero_grad(self):
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()
        self.opt_s.zero_grad()

    def step_opts(self):
        self.opt_c1.step()
        self.opt_c2.step()
        self.opt_s.step()
        
    def to(self, device):
        self.client_model_1.to(device)
        self.client_model_2.to(device)
        self.server.to(device)