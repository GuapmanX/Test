import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os.path as path

training_data_batch_size = 10

class MAB(nn.Module):
    def __init__(self, input_size, hidden_size,learning_rate):
        super().__init__()
        self.network =  nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.Iterations = 0

    #def __epsilonGreedy(self,X, totalIterations, Iteration):
        #epsilon = np.clip(Iteration/totalIterations,0,0.95)
        #if random.randint(1,100)/100 < epsilon:
            #exploitation
        #return self.network(X)
        #else:
            #exploration
            #return torch.rand(training_data_batch_size, 3, requires_grad=True)

    def forward(self,X,MaxIterations):
        self.Iterations +=1
        return self.network(X) #self.__epsilonGreedy(X,MaxIterations,self.Iterations)
    
    def backward(self,Y_prediction,Y):
        loss = self.criterion(Y_prediction, Y)
        loss.backward()

    def optimize(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def predict(self,X,MaxIterations):
        return  torch.argmax(self.network(X)).item() #torch.argmax(self.__epsilonGreedy(X,MaxIterations,self.Iterations)).item()
        

def GenerateTrainingData(input_size,batches):
    Data = torch.rand(batches,input_size)
    Largest = torch.argmax(Data, dim=1)
    return Data, Largest

MAX_ITERATIONS = 200

FILE = "model.pth"

model = MAB(3,9,0.01)

if path.exists(FILE):
    model.load_state_dict(torch.load(FILE))
    model.eval()

newdata = torch.tensor([364.0,790.0,889.0])
print(f'before: {model.predict(newdata,200)}')

for i in range(MAX_ITERATIONS):
    engagement, correct = GenerateTrainingData(3, training_data_batch_size)

    prediction = model.forward(engagement,MAX_ITERATIONS)
    model.backward(prediction,correct)
    model.optimize()

print(f'after: {model.predict(newdata,200)}')


torch.save(model.state_dict(),FILE)