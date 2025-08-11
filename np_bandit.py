import numpy as np

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

arms = 3
epsilon = 0.1
iterations = 100
rewards = [100.0, 200.0, 230.0]

agent = MultiArmedBandit(3, epsilon)

#print(np.random.randn(arms, iterations))
print(f'before training = {agent.selectArm()}')

for t in range(iterations):
    arm = agent.selectArm()
    reward = rewards[arm]
    agent.update(arm,reward)

print(f'after training = {agent.selectArm()}')
