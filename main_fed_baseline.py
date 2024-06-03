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

from utils.sampling import mnist_iid, mnist_noniid, split_VFL_data, cifar_iid,cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import ClientModelmnist, ServerModelmnist, VHFLGroup, VHFLGroupCifar
from models.Fed import FedAvg
from models.test import test_img
# from efficientnet_pytorch import EfficientNet
from data.imageVFL import imageVFL, SplitImageDataset
import csv



def write_to_csv(file_name, training_loss, testing_loss, training_acc, testing_acc):
    # Parse
    rows = []
    for i in range(len(training_loss)):
        row = [i+1, training_loss[i], testing_loss[i], training_acc[i].item(), testing_acc[i].item()]
        rows.append(row)

    with open(file_name, 'w', newline='') as csvfile:
        fields = ['Communication_Round', 'Training_loss', 'Testing_loss', 'Training_Accuracy', 'Testing_Accuracy']
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

    print("Write_to_CSV_Complete")
    return 0

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    HFL_CLIENTS = args.HFL_Clients
    VFL_CLIENTS = args.VFL_Clients

    HFL_DATA = args.HFL_Data_Split
    VFL_DATA = 1 - HFL_DATA

    # load dataset and split users
    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        if args.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        elif args.dataset == 'fmnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.FashionMNIST('../data/fmnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.FashionMNIST('../data/fmnist/', train=False, download=True, transform=trans_mnist)
        # dataset_train = split_VFL_data
        img, label, _ = split_VFL_data(dataset_train)
        
        img_1 = torch.cat(img[0])  # get the first split of the data
        img_2 = torch.cat(img[1])  # get the second split of the data

        img_1 = img_1.view(img_1.shape[0], -1)
        img_2 = img_2.view(img_2.shape[0], -1)

        data_length = len(img_1)
        HFL_Split = int(HFL_DATA * data_length)


        # First 40000 images, HFL & non iid, 4 clients
        dataset_train_HFL_1 = img_1[:HFL_Split]
        dataset_train_HFL_2 = img_2[:HFL_Split]
        label_HFL = label[:HFL_Split]
        # Create dataset

        dataset_train_HFL = imageVFL(dataset_train_HFL_1, dataset_train_HFL_2, label_HFL)
        dict_users_HFL = mnist_noniid(dataset_train_HFL, HFL_CLIENTS)

        # Last 20000 images, VFL, 2 GROUPS of clients, iid
        dataset_train_VFL_1 = img_1[HFL_Split:]
        dataset_train_VFL_2 = img_2[HFL_Split:]
        label_VFL = label[HFL_Split:]
        dataset_train_VFL = imageVFL(dataset_train_VFL_1, dataset_train_VFL_2, label_VFL)
        dict_users_VFL = mnist_iid(dataset_train_VFL, VFL_CLIENTS)
        

        img_test, label_test, _ = split_VFL_data(dataset_test)
        img_test_1 = torch.cat(img_test[0])  # get the first split of the data
        img_test_2 = torch.cat(img_test[1])  # get the second split of the data
        img_test_1 = img_test_1.view(img_test_1.shape[0], -1)
        img_test_2 = img_test_2.view(img_test_2.shape[0], -1)
        dataset_test = imageVFL(img_test_1, img_test_2, label_test)

        # print(len(dataset_train_HFL))
        # print(len(dataset_train_VFL))
        # print(len(dataset_test))


    elif args.dataset == 'cifar':
        # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        # dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_test)

        img_1, img_2, label, _ = SplitImageDataset(dataset_train)
        
        # img_1 = torch.cat(img[0])  # get the first split of the data
        # img_2 = torch.cat(img[1])  # get the second split of the data



        # img_1 = img_1.view(img_1.shape[0], -1)
        # img_2 = img_2.view(img_2.shape[0], -1)

        # print(img_1.shape)
        data_length = len(img_1)
        HFL_Split = int(HFL_DATA * data_length)
        


        # First 30000 images, HFL & non iid, 3 clients
        dataset_train_HFL_1 = img_1[:HFL_Split]
        dataset_train_HFL_2 = img_2[:HFL_Split]
        label_HFL = label[:HFL_Split]
        # Create dataset

        dataset_train_HFL = imageVFL(dataset_train_HFL_1, dataset_train_HFL_2, label_HFL)
        dict_users_HFL = cifar_noniid(dataset_train_HFL, HFL_CLIENTS)

        # Last 20000 images, VFL, 2 GROUPS of clients, iid
        dataset_train_VFL_1 = img_1[HFL_Split:]
        dataset_train_VFL_2 = img_2[HFL_Split:]
        label_VFL = label[HFL_Split:]
        dataset_train_VFL = imageVFL(dataset_train_VFL_1, dataset_train_VFL_2, label_VFL)
        dict_users_VFL = cifar_iid(dataset_train_VFL, VFL_CLIENTS)
        

        img_test_1, img_test_2, label_test, _ = SplitImageDataset(dataset_test)
        dataset_test = imageVFL(img_test_1, img_test_2, label_test)
    elif args.dataset == 'SVHN':
        # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        

        # dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        # dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        dataset_train = datasets.SVHN('../data/SVHN', split= "train" , download=True, transform=transform_train)
        dataset_test = datasets.SVHN('../data/SVHN', split= "train" , download=True, transform=transform_test)

        img_1, img_2, label, _ = SplitImageDataset(dataset_train)
        
        # img_1 = torch.cat(img[0])  # get the first split of the data
        # img_2 = torch.cat(img[1])  # get the second split of the data



        # img_1 = img_1.view(img_1.shape[0], -1)
        # img_2 = img_2.view(img_2.shape[0], -1)

        # print(img_1.shape)

        # Shape: SVHN Train got 73,257 data points, we take first 70000 to avoid non-integer problem in data-splitting    

        img_1 = img_1[:70000]
        img_2 = img_2[:70000]
        label = label[:70000]

        data_length = len(img_1)
        HFL_Split = int(HFL_DATA * data_length)


       
        dataset_train_HFL_1 = img_1[:HFL_Split]
        dataset_train_HFL_2 = img_2[:HFL_Split]
        label_HFL = label[:HFL_Split]
        # Create dataset

        dataset_train_HFL = imageVFL(dataset_train_HFL_1, dataset_train_HFL_2, label_HFL)
        dict_users_HFL = cifar_noniid(dataset_train_HFL, HFL_CLIENTS)

        
        dataset_train_VFL_1 = img_1[HFL_Split:]
        dataset_train_VFL_2 = img_2[HFL_Split:]
        label_VFL = label[HFL_Split:]
        dataset_train_VFL = imageVFL(dataset_train_VFL_1, dataset_train_VFL_2, label_VFL)
        dict_users_VFL = cifar_iid(dataset_train_VFL, VFL_CLIENTS)
        

        img_test_1, img_test_2, label_test, _ = SplitImageDataset(dataset_test)
        dataset_test = imageVFL(img_test_1, img_test_2, label_test)

    else:
        exit('Error: unrecognized dataset')

        
    






    
    # Pause CIFAR

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



    
    # img_size = dataset_train[0][0].shape

    # build model
    if args.dataset == 'mnist' or args.dataset =='fmnist':
        net_glob = VHFLGroup(args.opt, args.lr)
    elif args.dataset == 'cifar' or args.dataset == "SVHN":
        net_glob = VHFLGroupCifar(args.opt, args.lr)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    acc_train = []

    evaluation_frequency = 1


    loss_test = []
    acc_test = []
    

    # if args.all_clients: 
    #     print("Aggregation over all clients")
    #     w_locals = [w_glob for i in range(args.num_users)]

    # w_locals = [w_glob for i in range(args.num_users)]
    w_locals_HFL = []
    w_locals_VFL = []
    for iter in range(1, args.epochs+1):
        loss_locals = []
        acc_locals = []
        # m = max(int(args.frac * args.num_users), 1)
        

        # m = args.num_users
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # HFL Stage
        for idx in range(HFL_CLIENTS):
            # Each Iteration, we only use parts of the data
            fraction = 1
            selected_idxs = np.random.choice(dict_users_HFL[idx], int(fraction * len(dict_users_HFL[idx])), replace = False)
            # local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local = LocalUpdate(args=args, dataset=dataset_train_HFL, idxs=selected_idxs)
            net = copy.deepcopy(net_glob)
            net.to(args.device)
            w, loss, acc = local.train(net=net)


            w_locals_HFL.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            acc_locals.append(copy.deepcopy(acc))

            
            

        # VFL Stage
        for idx in range(VFL_CLIENTS):
            # Each Iteration, we only use parts of the data

            fraction = 1
            selected_idxs = np.random.choice(dict_users_VFL[idx], int(fraction * len(dict_users_VFL[idx])), replace = False)
            # local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            local = LocalUpdate(args=args, dataset=dataset_train_VFL, idxs=selected_idxs)
            # w, loss, acc = local.train(net=copy.deepcopy(net_glob).to(args.device))
            net = copy.deepcopy(net_glob)
            net.to(args.device)
            w, loss, acc = local.train(net=net)

            w_locals_VFL.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            acc_locals.append(copy.deepcopy(acc))


        # update global weights
        # agg_HFL = FedAvg(w_locals_HFL)

        # agg_VFL = FedAvg(w_locals_VFL)

        # w_glob = FedAvg([agg_HFL, agg_VFL])


        # Baseline
        baseline = w_locals_HFL + w_locals_VFL
        w_glob = FedAvg(baseline)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        # print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        acc_avg = sum(acc_locals) / len(acc_locals)
        acc_train.append(acc_avg)


        # Evaluate the accuracy
        if iter % evaluation_frequency == 0:
            net_glob.eval()
            net_glob.to(args.device)
            acc, loss = test_img(net_glob, dataset_test, args)
            print("Testing accuracy: {:.2f}".format(acc))
            net_glob.train()

            loss_test.append(loss)
            acc_test.append(acc)
        

    # Write to CSV
    file_path = './results/fed_{}_{}_{}_{}_HFL{}_VFL{}.csv'.format(args.dataset, args.epochs, args.opt, args.lr, args.HFL_Clients, args.VFL_Clients)
    write_to_csv(file_path, loss_train, loss_test, acc_train, acc_test)

    
    # plot loss curve
    
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train, label = "Train Loss")
    plt.plot(range(len(loss_test)), loss_test, label = "Test Loss")
    plt.ylabel('Loss')
    plt.xlabel("Communication Rounds")
    plt.legend()
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

