import numpy as np
import matplotlib.pyplot as plt

#epsilonGreedy

class MultiArmedBandit():
    def __init__(self,n_arms,epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.pulledCount = np.zeros(n_arms)
    
    def selectArm(self,PercievedValues):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_arms)
        else:
            return np.argmax(PercievedValues)
        
    def update(self,chosen_arm,reward,PercievedValues,pulledCount):
        pulledCount[chosen_arm] += 1
        n = pulledCount[chosen_arm]
        value = PercievedValues[chosen_arm]
        PercievedValues[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward
        return (PercievedValues, pulledCount)