import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

model_data = torch.zeros(3)
engagement = torch.tensor([100.0,305.0,300.0])

model = nn.Linear(3,3)

learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iterations = 200

def epsilonGreedy(totalIterations,Iteration):
    epsilon = Iteration/totalIterations
    if random.randint(1,100)/100 < epsilon:
        #exploitation
        return model(model_data)
    else:
        #exploration
        return torch.rand(3, requires_grad=True) #torch.multinomial(probs, 1).item()

for iteration in range(iterations):
    prediction = epsilonGreedy(200,iteration) #model(model_data)
    loss = criterion(prediction, engagement)

    #backward pass
    loss.backward()
    print(torch.argmax(prediction).item())#selected value

    #if iteration == 100:
        #engagement[0] += 500

    #update
    optimizer.step()
    optimizer.zero_grad()

