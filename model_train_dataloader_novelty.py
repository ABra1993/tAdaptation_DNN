import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import random

# # set the seed
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True

class PairsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # sample
        idx1 = random.randint(0, len(self.data) - 1)
        idx2 = random.randint(0, len(self.data) - 1)
        idx3 = random.randint(0, len(self.data) - 1)
        idx4 = random.randint(0, len(self.data) - 1)

        img1, label1 = self.data[idx1]
        img2, label2 = self.data[idx2]
        img3, label3 = self.data[idx3]
        img4, label4 = self.data[idx4]
        
        return (img1, img2, img3, img4), (label1, label2, label3, label4)