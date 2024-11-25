# python main_fed.py --dataset mnist  --epochs 50  --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt SGD --lr 0.2 --HFL_Clients 18 --VFL_Clients 0 --HFL_Data_Split 1
# python main_fed.py --dataset mnist  --epochs 50  --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt Adam --lr 0.003 --HFL_Clients 18 --VFL_Clients 0 --HFL_Data_Split 1



# python main_fed.py --dataset fmnist  --epochs 50 --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt SGD --lr 0.2 --HFL_Clients 18 --VFL_Clients 0 --HFL_Data_Split 1
# python main_fed.py --dataset fmnist  --epochs 50 --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt Adam --lr 0.003 --HFL_Clients 18 --VFL_Clients 0 --HFL_Data_Split 1



# python main_fed.py --dataset cifar  --epochs 50 --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt SGD --lr 0.2 --HFL_Clients 18 --VFL_Clients 0 --HFL_Data_Split 1
# python main_fed.py --dataset cifar  --epochs 50 --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt SGD --lr 0.001 --HFL_Clients 18 --VFL_Clients 0 --HFL_Data_Split 1


# python main_fed.py --dataset SVHN  --epochs 50 --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt SGD --lr 0.2 --HFL_Clients 18 --VFL_Clients 0 --HFL_Data_Split 1
# python main_fed.py --dataset SVHN  --epochs 50 --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt SGD --lr 0.001 --HFL_Clients 18 --VFL_Clients 0 --HFL_Data_Split 1



# If GPU
# python main_fed.py --dataset mnist  --epochs 50  --local_ep 2 --local_bs 128 --gpu 0 --num_users 6 --bs 128 --opt SGD --lr 0.2 --HFL_Clients 8 --VFL_Clients 2 --HFL_Data_Split 0.8
# python main_fed.py --dataset mnist  --epochs 50  --local_ep 2 --local_bs 128 --gpu 0 --num_users 6 --bs 128 --opt Adam --lr 0.003 --HFL_Clients 8 --VFL_Clients 2 --HFL_Data_Split 0.8



# python main_fed.py --dataset fmnist  --epochs 50 --local_ep 2 --local_bs 128 --gpu 0 --num_users 6 --bs 128 --opt SGD --lr 0.2 --HFL_Clients 8 --VFL_Clients 2 --HFL_Data_Split 0.8
# python main_fed.py --dataset fmnist  --epochs 50 --local_ep 2 --local_bs 128 --gpu 0 --num_users 6 --bs 128 --opt Adam --lr 0.003 --HFL_Clients 8 --VFL_Clients 2 --HFL_Data_Split 0.8



# python main_fed.py --dataset cifar  --epochs 50 --local_ep 2 --local_bs 128 --gpu 0 --num_users 6 --bs 128 --opt SGD --lr 0.2 --HFL_Clients 8 --VFL_Clients 2 --HFL_Data_Split 0.8
# python main_fed.py --dataset cifar  --epochs 50 --local_ep 2 --local_bs 128 --gpu 0 --num_users 6 --bs 128 --opt SGD --lr 0.001 --HFL_Clients 8 --VFL_Clients 2 --HFL_Data_Split 0.8


# python main_fed.py --dataset SVHN  --epochs 50 --local_ep 2 --local_bs 128 --gpu 0 --num_users 6 --bs 128 --opt SGD --lr 0.2 --HFL_Clients 8 --VFL_Clients 2 --HFL_Data_Split 0.8
# python main_fed.py --dataset SVHN  --epochs 50 --local_ep 2 --local_bs 128 --gpu 0 --num_users 6 --bs 128 --opt SGD --lr 0.001 --HFL_Clients 8 --VFL_Clients 2 --HFL_Data_Split 0.8


# python main_fed_pure_HFL.py --dataset cifar  --epochs 50 --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt SGD --lr 0.2 --HFL_Clients 18 --VFL_Clients 0 --HFL_Data_Split 1
# python main_fed_pure_HFL.py --dataset cifar  --epochs 50 --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt Adam --lr 0.001 --HFL_Clients 18 --VFL_Clients 0 --HFL_Data_Split 1


# python main_fed_pure_HFL.py --dataset SVHN  --epochs 50 --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt SGD --lr 0.2 --HFL_Clients 18 --VFL_Clients 0 --HFL_Data_Split 1
# python main_fed_pure_HFL.py --dataset SVHN  --epochs 50 --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt Adam --lr 0.001 --HFL_Clients 18 --VFL_Clients 0 --HFL_Data_Split 1

