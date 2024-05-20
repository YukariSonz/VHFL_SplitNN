import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class imageVFL(Dataset):

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
    
    

# Split an image into 2 pieces
def SplitImageDataset(dataset):
    dataset_dict = []
    label_list = []
    index_list = []
    image_1 = []
    image_2 = []
    idx = 0
    for tensor, label in dataset:
        # print(tensor.shape)
        height = tensor.shape[1]//2
        tensor_left = tensor[:, 0: height, :]
        tensor_right = tensor[:, height:, : ]
        image_1.append(tensor_left)
        image_2.append(tensor_right)
        label_list.append(torch.Tensor([label]))
        index_list.append(torch.Tensor([idx]))
        
        idx += 1
    image_1 = torch.stack(image_1)
    image_2 = torch.stack(image_2)
    return image_1, image_2, label_list, index_list


        