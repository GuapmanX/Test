import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

model_data = torch.zeros(3)
optimizer = optim.SGD([model_data], lr=0.01)

engagement = torch.tensor([100.0,700.0,300.0], requires_grad=True)

