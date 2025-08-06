import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

model_data = torch.tensor(3,dtype=torch.float32)

rewards = torch.tensor([10.0,30.0,50.0],dtype=torch.float32)
max_val = torch.max(rewards)

model = nn.Linear(1,1)

learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iterations = 20


for iteration in range(iterations):
    prediction = model(model_data)
    
    index = torch.argmax(prediction).item()

    loss = (model_data[index] - torch.max(prediction)) ** 2 #-torch.mean(max_val - prediction)

    loss.backward()
    #update
    optimizer.step()
    optimizer.zero_grad()
