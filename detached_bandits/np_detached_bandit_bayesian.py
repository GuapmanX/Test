import numpy as np
import matplotlib.pyplot as plt
#Thompson sampling

class ThompsonSampling:
    def __init__(self, n_arms, tolerance):
        self.n_arms = n_arms
        self.tolerance = tolerance
        self.epsilon = 0.1

    def selectArm(self, successes, failures):
        sampled_values = np.random.beta(successes, failures)
        return np.argmax(sampled_values)

    def update(self, chosen_arm, reward, successes, failures):
        if reward > self.tolerance:
            successes[chosen_arm] += 1
        else:
            failures[chosen_arm] += 1
            
        return(successes, failures)