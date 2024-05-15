import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class mnistVFL(Dataset):

    def __init__(self, img1, img2, label):
        self.data = torch.stack((img1, img2), dim=1)
        self.train_labels = torch.Tensor(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        index = int(index)
        img1 = self.data[index][0]
        img2 = self.data[index][1]
        label = self.train_labels[index]
        return (img1, img2,  label)