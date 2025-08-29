import numpy as np
import random
from detached_bandits.np_detached_bandit_bayesian import ThompsonSampling

class User:
    def __init__(self):
        #user's preferences
        self.user_newsl_type = np.ones(2)  #np.random.randint(0, 100, 2) #photo, video

        self.user_newsl_type[0] *= 100 #preset newsletter


        #self.user_photo = np.random.randint(0, 100, 3)
        #self.user_video = np.random.randint(0, 100, 3)
        self.user_preferences = [ #photo, video
            np.random.randint(0, 100, 3),
            np.random.randint(0, 100, 3)
        ]

        #makes the most familiar choices better
        self.user_newsl_type[self.user_newsl_type.argmax()] *= 10
        self.user_preferences[0][self.user_preferences[0].argmax()] *= 10
        self.user_preferences[1][self.user_preferences[1].argmax()] *= 10

        #bandit data
        self.newsl_type_successes = np.ones(2, dtype=np.int32)
        self.newsl_type_failiures = np.ones(2, dtype=np.int32)


        self.pNv_successes = [ #photo, video
            np.ones(3, dtype=np.int32),
            np.ones(3, dtype=np.int32)
        ]

        self.pNv_failiures = [
            np.ones(3, dtype=np.int32),
            np.ones(3, dtype=np.int32)
        ]

    def interest_shift(self,photoDelta, videoDelta):
        self.user_newsl_type[0] = self.__clamp(self.user_newsl_type[0] + photoDelta,1,100)
        self.user_newsl_type[1] = self.__clamp(self.user_newsl_type[1] + videoDelta,1,100)

    def decide_newsletter_type(self,newsletter):
        REJECTION_CONSTANT = 40
        return random.choices([False, True], [REJECTION_CONSTANT,self.user_newsl_type[newsletter]], k=1)[0]
    
    def decide_email(self,newsletter,nl_success,email):
        if(nl_success):
            REJECTION_CONSTANT = 40
            return random.choices([False, True], [REJECTION_CONSTANT , self.user_preferences[newsletter][email]], k=1)[0]
        else:
            return 0

    
    def __clamp(self, n, min_value, max_value):
        return max(min_value, min(n, max_value))
    
    def watchtime(self, picked, newsletter, email):
        if(picked):
            ATTENTION_SPAN_CONSTANT = 5
            pref_val = self.user_preferences[newsletter][email] #self.Preferences[email]
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
        
    def buysomething(self, picked, watchtime, newsletter, email):
        if(picked):
            ATTENTION_SPAN_CONSTANT = 5
            WATCHTIME_REINFORCEMENT_CONSTANT = 0.5
            pref_val = self.user_preferences[newsletter][email]
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
    
    def calculate_newsletter(self, newsletter_guess): #seperate for the other bandit
        Newsletter = self.decide_newsletter_type(newsletter_guess)

        PICKED_NEWSLETTER_MULTIPLIER = 25

        return Newsletter * PICKED_NEWSLETTER_MULTIPLIER

    def calculate_reward(self, newsletter_success, newsletter, bandit_guess):
        Picked = self.decide_email(newsletter, newsletter_success, bandit_guess)
        WatchTime = self.watchtime(Picked, newsletter,bandit_guess)
        Earnings = self.buysomething(Picked,WatchTime, newsletter,bandit_guess)

        PICKED_EMAIL_MULTIPLIER = 50
        WATCH_TIME_MULTIPLIER = 3
        EARNINGS_MULTIPLIER = 10

        return  Picked * PICKED_EMAIL_MULTIPLIER + WatchTime * WATCH_TIME_MULTIPLIER + Earnings * EARNINGS_MULTIPLIER
    
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


users = Userbase(100)
single_user = User()

nl_picker = ThompsonSampling(2,24)
email_picker = ThompsonSampling(3,70)

training_iterations = 10
testing_iterations = 1

for i in range(users.userCount()):
    for i in range(training_iterations):
        guess = nl_picker.selectArm(users.Users[i].newsl_type_successes,users.Users[i].newsl_type_failiures)
        newsl_reward = users.Users[i].calculate_newsletter(guess)
        users.Users[i].newsl_type_successes, users.Users[i].newsl_type_failiures = nl_picker.update(guess, newsl_reward, users.Users[i].newsl_type_successes, users.Users[i].newsl_type_failiures)
        
        #print(users.Users[i].pNv_successes[guess],users.Users[i].pNv_failiures[guess])
        email_guess = email_picker.selectArm(users.Users[i].pNv_successes[guess],users.Users[i].pNv_failiures[guess])
        content_reward = users.Users[i].calculate_reward(newsl_reward, guess,email_guess)
        users.Users[i].pNv_successes[guess], users.Users[i].pNv_failiures[guess] = email_picker.update(email_guess, content_reward, users.Users[i].pNv_successes[guess], users.Users[i].pNv_failiures[guess])
        users.Users[i].interest_shift(-5,5)



#test phase
Correct_nl_guesses = 0
for i in range(users.userCount()):
    for i in range(testing_iterations):
        guess = nl_picker.selectArm(users.Users[i].newsl_type_successes,users.Users[i].newsl_type_failiures)
        newsl_reward = users.Users[i].calculate_newsletter(guess)

        email_guess = email_picker.selectArm(users.Users[i].pNv_successes[guess],users.Users[i].pNv_failiures[guess])
        content_reward = users.Users[i].decide_email(guess, newsl_reward, email_guess)
        Correct_nl_guesses += content_reward

print((Correct_nl_guesses/users.userCount()) * 100)