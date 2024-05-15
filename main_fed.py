#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, split_VFL_data
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import ClientModelmnist, ServerModelmnist, VHFLGroup
from models.Fed import FedAvg
from models.test import test_img
# from efficientnet_pytorch import EfficientNet
from data.mnistVFL import mnistVFL

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)

        
        # dataset_train = split_VFL_data
        img, label, _ = split_VFL_data(dataset_train)
        
        img_1 = torch.cat(img[0])  # get the first split of the data
        img_2 = torch.cat(img[1])  # get the second split of the data

        img_1 = img_1.view(img_1.shape[0], -1)
        img_2 = img_2.view(img_2.shape[0], -1)



        # First 40000 images, HFL & non iid, 4 clients
        dataset_train_HFL_1 = img_1[:40000]
        dataset_train_HFL_2 = img_2[:40000]
        label_HFL = label[:40000]
        # Create dataset

        dataset_train_HFL = mnistVFL(dataset_train_HFL_1, dataset_train_HFL_2, label_HFL)
        dict_users_HFL = mnist_noniid(dataset_train_HFL, 4)

        # Last 20000 images, VFL, 2 GROUPS of clients, iid
        dataset_train_VFL_1 = img_1[40000:]
        dataset_train_VFL_2 = img_2[40000:]
        label_VFL = label[40000:]
        dataset_train_VFL = mnistVFL(dataset_train_VFL_1, dataset_train_VFL_2, label_VFL)
        dict_users_VFL = mnist_iid(dataset_train_VFL, 2)
        


        img_test, label_test, _ = split_VFL_data(dataset_test)
        img_test_1 = torch.cat(img_test[0])  # get the first split of the data
        img_test_2 = torch.cat(img_test[1])  # get the second split of the data
        img_test_1 = img_test_1.view(img_test_1.shape[0], -1)
        img_test_2 = img_test_2.view(img_test_2.shape[0], -1)
        dataset_test = mnistVFL(img_test_1, img_test_2, label_test)



        # print(len(dataset_train_HFL))
        # print(len(dataset_train_VFL))
        # print(len(dataset_test))






    
    # Palse CIFAR

    # elif args.dataset == 'cifar':
    #     # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #     transform_train = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])

    #     transform_test = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ])

    #     # dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
    #     # dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
    #     dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
    #     dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_test)

    #     #
    #     if args.iid:
    #         dict_users = cifar_iid(dataset_train, args.num_users)
    #     else:
    #         dict_users = cifar_noniid(dataset_train, args.num_users)
    #         #exit('Error: only consider IID setting in CIFAR10')



    else:
        exit('Error: unrecognized dataset')
    # img_size = dataset_train[0][0].shape

    # build model
    net_glob = VHFLGroup()
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    evaluation_frequency = 1
    

    # if args.all_clients: 
    #     print("Aggregation over all clients")
    #     w_locals = [w_glob for i in range(args.num_users)]

    # w_locals = [w_glob for i in range(args.num_users)]
    w_locals_HFL = []
    w_locals_VFL = []
    for iter in range(1, args.epochs+1):
        loss_locals = []
        # m = max(int(args.frac * args.num_users), 1)
        

        # m = args.num_users
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # HFL Stage
        for idx in range(4):
            # Each Iteration, we only use parts of the data
            fraction = 1
            selected_idxs = np.random.choice(dict_users_HFL[idx], int(fraction * len(dict_users_HFL[idx])), replace = False)
            # local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local = LocalUpdate(args=args, dataset=dataset_train_HFL, idxs=selected_idxs)
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))


            w_locals_HFL.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

            
            

        # VFL Stage
        for idx in range(2):
            # Each Iteration, we only use parts of the data

            fraction = 1
            selected_idxs = np.random.choice(dict_users_VFL[idx], int(fraction * len(dict_users_VFL[idx])), replace = False)
            # local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local = LocalUpdate(args=args, dataset=dataset_train_VFL, idxs=selected_idxs)
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            w_locals_VFL.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))


        # update global weights
        agg_HFL = FedAvg(w_locals_HFL)

        agg_VFL = FedAvg(w_locals_VFL)

        w_glob = FedAvg([agg_HFL, agg_VFL])

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # Evaluate the accuracy
        if iter % evaluation_frequency == 0:
            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print("Testing accuracy: {:.2f}".format(acc_test))
            net_glob.train()
        
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        # print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing, freeup the memory
    # del(w_glob)
    # del(w_locals)
    # torch.cuda.empty_cache()

    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Training accuracy: {:.2f}".format(acc_train))
    # print("Testing accuracy: {:.2f}".format(acc_test))

