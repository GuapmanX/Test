import numpy as np
import random
from detached_bandits.np_detached_bandit import MultiArmedBandit
from detached_bandits.np_detached_bandit_bayesian import ThompsonSampling


class User():
    def __init__(self):
        self.Preferences = np.random.randint(1,100,3)
        self.Successes = np.ones(3)
        self.Failiures = np.ones(3)
        self.BanditPulls = np.array([0,0,0])
        self.Preferences[self.Preferences.argmax()] *= 5 #user is more likely to pick what they like so this just balances it out
    
    def decide_email(self,email):
        REJECTION_CONSTANT = 50
        return random.choices([0, 1], [REJECTION_CONSTANT , self.Preferences[email]], k=1)[0]
    
    def __clamp(self, n, min_value, max_value):
        return max(min_value, min(n, max_value))
    
    def watchtime(self, picked, email):
        if(bool(picked)):
            ATTENTION_SPAN_CONSTANT = 5
            pref_val = self.Preferences[email]
            return random.choices([0, 5, 10, 15, 20, 25, 30], [ #the further the video goes on, the more likely to click off
                self.__clamp(pref_val,1,99999999), 
                self.__clamp(pref_val - ATTENTION_SPAN_CONSTANT,1,99999999),
                self.__clamp(pref_val - ATTENTION_SPAN_CONSTANT * 2,1,99999999),
                self.__clamp(pref_val - ATTENTION_SPAN_CONSTANT * 3,1,99999999),
                self.__clamp(pref_val - ATTENTION_SPAN_CONSTANT * 4,1,99999999),
                self.__clamp(pref_val - ATTENTION_SPAN_CONSTANT * 5,1,99999999),
                self.__clamp(pref_val - ATTENTION_SPAN_CONSTANT * 6,1,99999999)
                ], k=1)[0]
        else:
            return 0
        
    def buysomething(self, picked, watchtime, email):
        if(bool(picked)):
            ATTENTION_SPAN_CONSTANT = 5
            WATCHTIME_REINFORCEMENT_CONSTANT = 0.5
            pref_val = self.Preferences[email]
            return random.choices([0, 5, 10, 15, 20, 25, 30], [ #the more expensive, the less likely
                self.__clamp(pref_val,1,99999999), 
                self.__clamp(pref_val - ATTENTION_SPAN_CONSTANT + watchtime * WATCHTIME_REINFORCEMENT_CONSTANT,1,99999999),
                self.__clamp(pref_val - ATTENTION_SPAN_CONSTANT * 2 + watchtime * WATCHTIME_REINFORCEMENT_CONSTANT * 2,1,99999999),
                self.__clamp(pref_val - ATTENTION_SPAN_CONSTANT * 3 + watchtime * WATCHTIME_REINFORCEMENT_CONSTANT * 3,1,99999999),
                self.__clamp(pref_val - ATTENTION_SPAN_CONSTANT * 4 + watchtime * WATCHTIME_REINFORCEMENT_CONSTANT * 4,1,99999999),
                self.__clamp(pref_val - ATTENTION_SPAN_CONSTANT * 5 + watchtime * WATCHTIME_REINFORCEMENT_CONSTANT * 5,1,99999999),
                self.__clamp(pref_val - ATTENTION_SPAN_CONSTANT * 6 + watchtime * WATCHTIME_REINFORCEMENT_CONSTANT * 6,1,99999999)
                ], k=1)[0]
        else:
            return 0
    
    def calculate_reward(self, bandit_guess):
        Picked = self.decide_email(bandit_guess)
        WatchTime = self.watchtime(Picked,bandit_guess)
        Earnings = self.buysomething(Picked,WatchTime,bandit_guess)

        PICKED_EMAIL_MULTIPLIER = 50
        WATCH_TIME_MULTIPLIER = 3
        EARNINGS_MULTIPLIER = 10

        return Picked * PICKED_EMAIL_MULTIPLIER + WatchTime * WATCH_TIME_MULTIPLIER + Earnings * EARNINGS_MULTIPLIER




class Userbase():
    def __init__(self,BaseAmount):
        self.Users = []
        self.GeneralBanditData = np.array([1,1,1])
        self.Successes = np.array([1,1,1])
        self.Failiures = np.array([1,1,1])
        for i in range(BaseAmount):
            self.register(False)

    def register(self, AddAverageData):
        value = User()

        if(AddAverageData):
            self.recalculateAverageBandit()
            value.BanditData = self.GeneralBanditData

        self.Users.append(value)
        return len(self.Users) - 1

    def userCount(self):
        return len(self.Users)

    def recalculateAverageBandit(self):
        TotalSucceses = np.array([0,0,0])
        TotalFailiures = np.array([0,0,0])
        Users = self.userCount()
        for i in range(Users):
            np.add(TotalSucceses,self.Users[i].Successes)
            np.add(TotalFailiures,self.Users[i].Failiures)
        self.Successes = TotalSucceses/Users
        self.Failiures = TotalFailiures/Users



base = Userbase(100)
Agent = ThompsonSampling(3,70)
testIterations = 5
    

#training phase
for i in range(base.userCount()):
    for epoch in range(testIterations):
        bandit_guess = Agent.selectArm(base.Users[i].Successes, base.Users[i].Failiures)


        reward = base.Users[i].calculate_reward(bandit_guess)
        base.Users[i].Successes, base.Users[i].Failiures = Agent.update(bandit_guess, reward, base.Users[i].Successes, base.Users[i].Failiures)



correct_guesses = 0
#test phase
for i in range(base.userCount()):
        bandit_guess = Agent.selectArm(base.Users[i].Successes, base.Users[i].Failiures)
        chose = base.Users[i].decide_email(bandit_guess)
        correct_guesses += chose
        
print(f"Correct guesses = {correct_guesses}")

corret_newuser_guesses = 0
#test for newly registered users
for i in range(100):
    idx = base.register(True)
    bandit_guess = Agent.selectArm(base.Users[idx].Successes, base.Users[idx].Failiures)
    chose = base.Users[idx].decide_email(bandit_guess)
    corret_newuser_guesses += chose

print(f"Correct new user guesses = {corret_newuser_guesses}")