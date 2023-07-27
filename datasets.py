import numpy as np
# import torch
from torch.utils.data import Dataset #, DataLoader
from shapes import random_rectangle #, plot_result


class RectangleDataset(Dataset):
    def __init__(self, n_size, xy_size):
        """Torch Dataset with n_size images of size xy_size with 1 random rectangle """
        self.n_size = n_size
        self.xy_size = xy_size
        self.data = [random_rectangle(full_size=self.xy_size) for i in range(self.n_size)]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = np.expand_dims(self.data[idx][0], axis=0)
        label = self.data[idx][1]
        
        return img, np.array(label)

