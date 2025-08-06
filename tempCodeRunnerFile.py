import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

data = [1, 2, 3, 4, 5]

tensor = torch.tensor(data)

print(tensor)

print(f"Tensor shape: {tensor.shape}")

class MailDataSet(Dataset):
    def __init__(self, data_dir, transform=None):
        pass
    
    def __len__(self):
        pass

    def __getitem__(self, idx):
        return 