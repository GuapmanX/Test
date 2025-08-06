import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


    
preferences = torch.zeros(3, requires_grad=True)
optimizer = optim.SGD([preferences], lr=0.1)

#probabilites = torch.tensor([ 0.1, 0.5, 0.25 ])
probs = torch.tensor([0.1,0.1,0.3],requires_grad=True)
payouts = [ 500.0, 100.0, 300.0 ]

iterations = 100

print(np.max(payouts))

def epsilonGreedy(totalIterations,Iteration):
    epsilon = Iteration/totalIterations
    if random.randint(1,100)/100 < epsilon:
        #exploitation
        return torch.argmax(probs).item()
    else:
        #exploration
        return torch.multinomial(probs, 1).item()

for epoch in range(iterations):
    action = epsilonGreedy(iterations,epoch) #torch.multinomial(probs, 1).item()
    reward = torch.tensor(payouts[action], dtype=torch.float32)
    
    loss = -torch.log(probs[action]) * reward
    print(payouts[action], loss)
    loss.backward()


    optimizer.step()
    optimizer.zero_grad()
