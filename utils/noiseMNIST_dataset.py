from torch.utils.data import Dataset, DataLoader

class noiseMNIST_dataset(Dataset):

    def __init__(self, noise_imgs, noise_lbls):

        self.imgs = noise_imgs
        self.lbls = noise_lbls

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # return (self.imgs[index, 0, :, :, :], self.imgs[index, 1, :, :, :], self.imgs[index, 2, :, :, :], self.lbls[index])
        return (self.imgs[index, :, :, :, :], self.lbls[index])