import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class MAB(nn.Module):
    def __init__(self, input_size, output_size,learning_rate):
        super().__init__()
        self.linear = nn.Linear(input_size,output_size)
        self.criterion = nn.MSELoss()
        self.reference = torch.ones(3)
        self.optimizer = torch.optim.SGD(self.linear.parameters(), lr=learning_rate)
        self.Iterations = 0

    def __epsilonGreedy(self, totalIterations, Iteration):
        epsilon = np.clip(Iteration/totalIterations,0,0.95)
        if random.randint(1,100)/100 < epsilon:
            #exploitation
            return self.linear(self.reference)
        else:
            #exploration
            return torch.rand(3, requires_grad=True)

    def forward(self,X,MaxIterations):
        self.Iterations +=1
        return self.__epsilonGreedy(MaxIterations,self.Iterations)
    
    def backward(self,Y_prediction,Y):
        loss = self.criterion(Y_prediction, Y)
        loss.backward()

    def optimize(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def predict(self,MaxIterations):
        return torch.argmax(self.__epsilonGreedy(MaxIterations,self.Iterations)).item()
        
engagement = torch.tensor([100.0,305.0,300.0])
MAX_ITERATIONS = 200

model = MAB(3,3,0.01)

for i in range(MAX_ITERATIONS):
    prediction = model.forward(MAX_ITERATIONS)
    model.backward(prediction,engagement)
    model.optimize()


print(model.predict(MAX_ITERATIONS))