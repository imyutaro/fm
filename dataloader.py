import os
import math
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

class BFM_Dataset(IterableDataset):
    """
    Load and convert dataset from tuple to pytorch tensor
    for basket factorization machine.

    Variables ----------
    --------------------
    """

    def __init__(self, filename="train.pkl", root_dir="./data/ta_feng/", seed=1234):
        super(BFM_Dataset).__init__()
        self.root_dir = root_dir
        self.filepath = os.path.join(self.root_dir, filename)

        f = open(self.filepath, "rb")
        self.data = pickle.load(f)
        # Shuffled by myself but IterableDataset has shuffle option,
        # so we don't need shuffle by myself.
        # data = random.Random(seed).sample(data, len(data))

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return iter()
        else:
            per_worker = len(data) // batch_size
            worker_id = worker_info.id
            return iter()

    def __len__(self):
        return len(self.data)

    def n_item(self):
        prod_set = set()
        for i in self.data:
            prod_set |= i[1]
        return len(prod_set)


if __name__=="__main__":
    ds = BFM_Dataset()

    data = DataLoader(ds, num_workers=20)
    print(data)

