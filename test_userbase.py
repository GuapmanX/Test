import numpy as np
import random
from detached_bandits.np_detached_bandit import MultiArmedBandit


class User():
    def __init__(self):
        self.Preferences = np.random.randint(1,100,3)
        self.BanditData = np.array([1,1,1])
        self.BanditPulls = np.array([0,0,0])
        self.Preferences[self.Preferences.argmax()] *= 5 #user is more likely to pick what they like so this just balances it out

    def select_Email(self):
        return random.choices([0, 1, 2], weights=self.Preferences, k=1)[0]
    
    def decide_email(self,email):
        REJECTION_CONSTANT = 50
        return random.choices([0, 1], [REJECTION_CONSTANT , self.Preferences[email]], k=1)[0]
    
    




class Userbase():
    def __init__(self,BaseAmount):
        self.Users = []
        self.GeneralBanditData = np.array([1,1,1])
        for i in range(BaseAmount):
            self.register()

    def register(self):
        value = User()
        self.Users.append(value)

    def userCount(self):
        return len(self.Users)

    def recalculateAverageBandit(self):
        Total = np.array([1,1,1])
        Users = self.userCount()
        for i in range(Users):
            Total += self.Users[i].BanditData
        self.GeneralBanditData = Total/Users



base = Userbase(100)
Agent = MultiArmedBandit(3,0.1)
testIterations = 50

def CalculateReward(correct_guess,bandit_guess):
    reward = 0

    if(bandit_guess == correct_guess):
        reward += 50
    
    return reward
    

#training phase
for i in range(base.userCount()):
    for epoch in range(testIterations):
        correct_value = base.Users[i].select_Email()
        bandit_guess = Agent.selectArm(base.Users[i].BanditData)
        base.Users[i].watched_Course(bandit_guess)
        reward = CalculateReward(correct_value, bandit_guess)
        base.Users[i].BanditData, base.Users[i].BanditPulls = Agent.update(bandit_guess, reward, base.Users[i].BanditData, base.Users[i].BanditPulls)

randomI = np.random.randint(0,base.userCount())
user_email = base.Users[randomI].select_Email()
bandit_guess = Agent.selectArm(base.Users[randomI].BanditData)

correct_guesses = 0
#test phase
for i in range(base.userCount()):
        correct_value = base.Users[i].select_Email()
        bandit_guess = Agent.selectArm(base.Users[i].BanditData)
        if(correct_value == bandit_guess):
            correct_guesses += 1
        
print(f"Correct guesses = {correct_guesses}")
        