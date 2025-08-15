import numpy as np
import matplotlib.pyplot as plt

#Thompson sampling + epsilon

class ThompsonSampling:
    def __init__(self, n_arms, tolerance, epsilon):
        self.n_arms = n_arms
        self.successes = np.ones(n_arms)
        self.failures = np.ones(n_arms)
        self.tolerance = tolerance
        self.epsilon = epsilon

    def selectArm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_arms)
        else:
            sampled_values = np.random.beta(self.successes + 1, self.failures + 1)
            return np.argmax(sampled_values)

        #sampled_values = np.random.beta(self.successes, self.failures)
        #return np.argmax(sampled_values)

    def update(self, chosen_arm, reward):
        if reward > self.tolerance:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1