python main_fed.py --dataset mnist  --epochs 50  --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt SGD --lr 0.2
python main_fed.py --dataset mnist  --epochs 50  --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt Adam --lr 0.003



python main_fed.py --dataset fmnist  --epochs 50  --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt SGD --lr 0.2
python main_fed.py --dataset fmnist  --epochs 50 --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt Adam --lr 0.003



# python main_fed.py --dataset cifar  --epochs 50 --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt SGD --lr 0.2
# python main_fed.py --dataset cifar  --epochs 50 --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt SGD --lr 0.001


# python main_fed.py --dataset SVHN  --epochs 50 --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt SGD --lr 0.2
# python main_fed.py --dataset SVHN  --epochs 50  --local_ep 2 --local_bs 128 --gpu -1 --num_users 6 --bs 128 --opt SGD --lr 0.001