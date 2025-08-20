import numpy as np
import random
from detached_bandits.np_detached_bandit import MultiArmedBandit

#TastePattern = [[6,[3.0,0.6,0.3]], [12,[1.0,4.0,0.3]], [24,[0.25,1.25,3.65]]] # months, pattern
TastePattern = [
    [3,np.random.randint(0,5,3)], [6,np.random.randint(0,5,3)], [9,np.random.randint(0,5,3)],
    [12,np.random.randint(0,5,3)], [15,np.random.randint(0,5,3)], [18,np.random.randint(0,5,3)],
    [21,np.random.randint(0,5,3)], [24,np.random.randint(0,5,3)]] # months, pattern


class User():
    def __init__(self):
        self.Preferences = np.random.randint(1,100,3)
        self.AccountAge = random.randint(0,50)

        #makes the user's taste a bit more predictable, based on their account age
        for i in range(len(TastePattern)):
            if(TastePattern[i][0] > self.AccountAge):
                    np.multiply(self.Preferences, TastePattern[i][1])

        self.BanditData = np.array([1,1,1])
        self.BanditPulls = np.array([0,0,0])
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
        #self.GeneralBanditData = np.array([1,1,1])
        self.GeneralBanditData = [
            [3,[0.0,0.0,0.0]], [6,[0.0,0.0,0.0]], [9,[0.0,0.0,0.0]],
            [12,[0.0,0.0,0.0]], [15,[0.0,0.0,0.0]], [18,[0.0,0.0,0.0]],
            [21,[0.0,0.0,0.0]], [24,[0.0,0.0,0.0]]
        ]

        for i in range(BaseAmount):
            self.register(False)

    def register(self, AddAverageData):
        value = User()

        if(AddAverageData):
            self.recalculateAverageBandit()

            #finds the best model based on account age
            for i in range(len(self.GeneralBanditData)):
                if(self.GeneralBanditData[i][0] > value.AccountAge):
                    value.BanditData = self.GeneralBanditData[i][1]

        self.Users.append(value)
        return len(self.Users) - 1

    def userCount(self):
        return len(self.Users)

    def recalculateAverageBandit(self):
        Total = [
            [3,[0.0,0.0,0.0],1], [6,[0.0,0.0,0.0],1], [9,[0.0,0.0,0.0],1],
            [12,[0.0,0.0,0.0],1], [15,[0.0,0.0,0.0],1], [18,[0.0,0.0,0.0],1],
            [21,[0.0,0.0,0.0],1], [24,[0.0,0.0,0.0],1]
        ] # months, pattern
        Users = self.userCount()

        #splits newer and older user training models
        for i in range(Users):
            taste_idx = 0
            for Z in range(len(Total)):
                if(Total[Z][0] > self.Users[i].AccountAge):
                    taste_idx = Z
            Total[taste_idx][1] += self.Users[i].BanditData
            Total[taste_idx][2] += 1

        #calculates average bandit for the appropriate model
        for i in range(len(self.GeneralBanditData)):
            for z in range(len(self.GeneralBanditData[i][1])):
                self.GeneralBanditData[i][1][z] = Total[i][1][z] / Total[i][2]



base = Userbase(100)
Agent = MultiArmedBandit(3,0.1)
testIterations = 50
    

#training phase
for i in range(base.userCount()):
    for epoch in range(testIterations):
        bandit_guess = Agent.selectArm(base.Users[i].BanditData)


        reward = base.Users[i].calculate_reward(bandit_guess)
        base.Users[i].BanditData, base.Users[i].BanditPulls = Agent.update(bandit_guess, reward, base.Users[i].BanditData, base.Users[i].BanditPulls)



correct_guesses = 0
#test phase
for i in range(base.userCount()):
        bandit_guess = Agent.selectArm(base.Users[i].BanditData)
        chose = base.Users[i].decide_email(bandit_guess)
        correct_guesses += chose
        
print(f"Correct guesses = {correct_guesses}")

corret_newuser_guesses = 0
#test for newly registered users
for i in range(100):
    idx = base.register(True)
    bandit_guess = Agent.selectArm(base.Users[idx].BanditData)
    chose = base.Users[idx].decide_email(bandit_guess)
    corret_newuser_guesses += chose

print(f"Correct new user guesses = {corret_newuser_guesses}")