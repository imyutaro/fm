import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class fm_data(Dataset):
    """
    Load dataset for factorization machine.

    Variables ----------
    --------------------
    """

    def __init__(self, filename="transaction.dat", root_dir="./data/"):
        self.root_dir = root_dir
        self.filepath = os.path.join(self.root_dir, filename)

        f = open(filename,"rb")
        data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass
