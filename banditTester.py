from np_bandit import MultiArmedBandit
from np_bandit_bayesian import ThompsonSampling
from np_UCB import UCB

import numpy as np
import matplotlib.pyplot as plt


n_arms = 10
iterations = 100
rewards = np.random.randint(0,100,n_arms)

#thompson
bayesian = ThompsonSampling(n_arms,75)
bayesian_wallet = 0.0
bayesian_data = np.zeros(iterations)

#epsilon
epsilon = MultiArmedBandit(n_arms,0.1)
epsilon_wallet = 0.0
epsilon_data = np.zeros(iterations)

#UCB
ucb = UCB(n_arms)
ucb_wallet = 0.0
ucb_data = np.zeros(iterations)

#best result
best_wallet = 0.0
best_data = np.zeros(iterations)

for t in range(iterations):

    #thompson
    bayesian_arm = bayesian.selectArm()
    bayesian_reward = rewards[bayesian_arm]
    bayesian.update(bayesian_arm,bayesian_reward)

    bayesian_wallet += bayesian_reward
    bayesian_data[t] = bayesian_wallet

    #epsilon
    epsilon_arm = bayesian.selectArm()
    epsilon_reward = rewards[epsilon_arm]
    epsilon.update(epsilon_arm,epsilon_reward)

    epsilon_wallet += epsilon_reward
    epsilon_data[t] = epsilon_wallet

    #ucb
    ucb_arm = bayesian.selectArm()
    ucb_reward = rewards[ucb_arm]
    ucb.update(ucb_arm,ucb_reward)

    ucb_wallet += ucb_reward
    ucb_data[t] = ucb_wallet

    #best result
    best_wallet += rewards[np.argmax(rewards)]
    best_data[t] = best_wallet


print(f"epsilon final result = {epsilon_data[epsilon_data.size - 1]}")
print(f"thompson final result = {bayesian_data[bayesian_data.size - 1]}")
print(f"ucb final result = {ucb_data[ucb_data.size - 1]}")
print(f"most optimal result = {best_data[best_data.size - 1]}")
plt.plot(bayesian_data,'b')
plt.plot(epsilon_data,'r')
plt.plot(ucb_data,'g')
plt.plot(best_data,'y')

plt.show()
