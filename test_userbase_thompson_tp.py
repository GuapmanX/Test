import numpy as np
import random
from detached_bandits.np_detached_bandit import MultiArmedBandit
from detached_bandits.np_detached_bandit_bayesian import ThompsonSampling

TastePattern = [
    [3,np.random.randint(0,5,3)], [6,np.random.randint(0,5,3)], [9,np.random.randint(0,5,3)],
    [12,np.random.randint(0,5,3)], [15,np.random.randint(0,5,3)], [18,np.random.randint(0,5,3)],
    [21,np.random.randint(0,5,3)], [24,np.random.randint(0,5,3)]] # months, pattern

class User():
    def __init__(self):
        self.Preferences = np.random.randint(1,100,3)
        self.Successes = np.ones(3)
        self.Failiures = np.ones(3)
        self.AccountAge = random.randint(0,50)

        for i in range(len(TastePattern)):
            if(TastePattern[i][0] > self.AccountAge):
                np.multiply(self.Preferences, TastePattern[i][1])


        self.Preferences[self.Preferences.argmax()] *= 50 #user is more likely to pick what they like so this just balances it out
    
    def decide_email(self,email):
        REJECTION_CONSTANT = 40
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

        self.gSnF = [ #general successes and failiures (1st is success, 2nd is failiure)
            [3,[np.ones(3),np.ones(3)]], [6,[np.ones(3),np.ones(3)]], [9,[np.ones(3),np.ones(3)]],
            [12,[np.ones(3),np.ones(3)]], [15,[np.ones(3),np.ones(3)]], [18,[np.ones(3),np.ones(3)]],
            [21,[np.ones(3),np.ones(3)]], [24,[np.ones(3),np.ones(3)]]
        ]

        self.Successes = np.array([1,1,1])
        self.Failiures = np.array([1,1,1])

        for i in range(BaseAmount):
            self.register(False)

    def register(self, AddAverageData):
        value = User()

        if(AddAverageData):
            self.recalculateAverageBandit()
            #value.BanditData = self.GeneralBanditData
            for i in range(len(self.gSnF)):
                if(self.gSnF[i][0] > value.AccountAge):
                    value.Successes = self.gSnF[i][1][0]
                    value.Failiures = self.gSnF[i][1][1]

        self.Users.append(value)
        return len(self.Users) - 1

    def userCount(self):
        return len(self.Users)

    def recalculateAverageBandit(self):
        NEWgSnF = [ #general successes and failiures (1st is success, 2nd is failiure)
            [3,[np.ones(3),np.ones(3)], 1], [6,[np.ones(3),np.ones(3)], 1], [9,[np.ones(3),np.ones(3)], 1],
            [12,[np.ones(3),np.ones(3)], 1], [15,[np.ones(3),np.ones(3)], 1], [18,[np.ones(3),np.ones(3)], 1],
            [21,[np.ones(3),np.ones(3)], 1], [24,[np.ones(3),np.ones(3)], 1]
        ]

        Users = self.userCount()

        for i in range(Users):
            taste_idx = 0
            for Z in range(len(NEWgSnF)):
                if(NEWgSnF[Z][0] > self.Users[i].AccountAge):
                    taste_idx = Z
            NEWgSnF[taste_idx][1][0] += self.Users[i].Successes
            NEWgSnF[taste_idx][1][1] += self.Users[i].Failiures
            NEWgSnF[taste_idx][2] += 1

            for i in range(len(self.gSnF)):
                for z in range(len(self.gSnF[i][1])):
                    self.gSnF[i][1][z] = np.round(NEWgSnF[i][1][z] / NEWgSnF[i][2])



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