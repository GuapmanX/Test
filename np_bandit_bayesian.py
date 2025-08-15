import numpy as np
import matplotlib.pyplot as plt

#Thompson sampling

class ThompsonSampling:
    def __init__(self, n_arms, tolerance):
        self.n_arms = n_arms
        self.successes = np.ones(n_arms)
        self.failures = np.ones(n_arms)
        self.tolerance = tolerance
        self.epsilon = 0.1

    def selectArm(self):
        sampled_values = np.random.beta(self.successes, self.failures)
        return np.argmax(sampled_values)

    def update(self, chosen_arm, reward):
        if reward > self.tolerance:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1