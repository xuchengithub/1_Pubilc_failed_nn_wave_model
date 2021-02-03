import numpy as np
import torch
from torch.utils.data import Dataset



class data_set_for_train(Dataset):
    def __init__(
        self,
        fold_rebuild_csv="./train_data/data_used_for_train_model_all.npy",
    ):
        # data loading
        a = list()
        b = list()
        xy = np.load(fold_rebuild_csv, allow_pickle=True)  #

        for i in range(len(xy)):
            a.append(xy[i, 0])
            b.append(xy[i, 1])
        a = np.array(a)
        b = np.array(b)
        # self.x = torch.from_numpy(a, dtype=torch.long)
        self.x = torch.tensor(a)
        # self.x = self.x.type(torch.LongTensor)
        # self.y = torch.from_numpy(b, dtype=torch.long)

        self.y = torch.tensor(b, dtype=torch.long)
        # self.y = self.y.type(torch.LongTensor)
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.n_samples
