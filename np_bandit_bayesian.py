import numpy as np
import matplotlib.pyplot as plt

#Thompson sampling

class ThompsonSampling:
    def __init__(self, n_arms, tolerance):
        self.n_arms = n_arms
        self.successes = np.zeros(n_arms)
        self.failures = np.zeros(n_arms)
        self.tolerance = tolerance

    def selectArm(self):
        sampled_values = np.random.beta(self.successes + 1, self.failures + 1)
        return np.argmax(sampled_values)

    def update(self, chosen_arm, reward):
        if reward > self.tolerance:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1