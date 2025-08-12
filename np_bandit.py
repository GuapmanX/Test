import numpy as np
import matplotlib.pyplot as plt

class MultiArmedBandit():
    def __init__(self,n_arms,epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.pulledCount = np.zeros(n_arms)
        self.PercievedValues = np.zeros(n_arms)
    
    def selectArm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_arms)
        else:
            return np.argmax(self.PercievedValues)
        
    def update(self,chosen_arm,reward):
        self.pulledCount[chosen_arm] += 1
        n = self.pulledCount[chosen_arm]
        value = self.PercievedValues[chosen_arm]
        self.PercievedValues[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward