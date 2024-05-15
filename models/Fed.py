#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg



# def FedAvg(w):
#     w_avg = copy.deepcopy(w[0])
#     for sub_model in w_avg:
#         for k in sub_model.keys():
#             for i in range(1, len(w)):
#                 w_avg[k] += w[i][k]
#             w_avg[k] = torch.div(w_avg[k], len(w))
#     return w_avg

# def FedAVGHFL(w):
#     w_avg = copy.deepcopy(w[0])
#     for i in range(1, len(w)):



# def FedAVGVFL(w):


# def FedAVGVHFL(w):