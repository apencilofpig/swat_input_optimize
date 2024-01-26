import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import struct

class SWaT_Dataset(Dataset):
    def __init__(self, np_inputs, np_labels):
        # self.df = df
        # self.inputs = df.iloc[:, :-1].values
        # self.labels = df.iloc[:, -1].values
        self.inputs = np_inputs
        self.labels = np_labels

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        input = self.inputs[index]
        label = self.labels[index]

        input = torch.tensor(input, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return input, label
    