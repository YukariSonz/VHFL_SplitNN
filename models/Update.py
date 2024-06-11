#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        img1, img2, label = self.dataset[self.idxs[item]]
        return img1, img2, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        # self.loss_func = nn.NLLLoss()
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        # train and update
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        correct = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_correct = 0

            for batch_idx, (img1, img2, labels) in enumerate(self.ldr_train):
                labels = labels.long()
                img1, img2,  labels = img1.to(self.args.device), img2.to(self.args.device),  labels.to(self.args.device)

                net.zero_grad()
                log_probs = net(img1, img2)
                loss = self.loss_func(log_probs, labels)
                loss.backward()


                net.step_opts()

                y_pred = log_probs.data.max(1, keepdim=True)[1]
                batch_correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum() / list(labels.shape)[0]
                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())

            correct.append(batch_correct/len(batch_loss))
            epoch_loss.append(sum(batch_loss)/len(batch_loss))


            local_loss = sum(batch_loss)/len(batch_loss)
            local_correct = batch_correct/len(batch_loss)

        
        # return net.state_dict(), sum(epoch_loss) / len(epoch_loss), 100 * sum(correct) / len(epoch_loss)
        return net.state_dict(), local_loss, local_correct
