#an agent is trained with reward-based algorithm to be able to control 
#a simple particle in 1D and avoid collision with an object o
#
#the same agent of greedy_agent_moving1D.py is used here,
#but game changes slightly:
#
#the state is computed calculating relative distance between particle X and o
#a higher reward is obtained if the particle is in a given range from the
#object (without colliding)
#
#s = 4: collision -> endgame reward -20
#s = 3: relative distance d <= 3 -> reward +5
#s = 2: relative distance -3 <= d reward +5
#s = 1: relative distance d > 0 -> reward +1
#s = 0: relative distance d < 0 -> reward +1
#
#the agent can choose to perform 2 actions
#a = 0: vx = -1
#a = 1: vx = +1
#
#using a q-learning algorithm the agent is able to win 100% of the evaluation
#episodes
#
#calculating the average reward per episode and the average reward per step
#some interesting informations can be extracted:
#1 - few training episodes are necessary to learn a policy that allows winning 
#    of 100% of evalaution episodes
#2 - increasing the number of maximum step in an episodes does not affect 
#    significantly the ratio of games won 
#3 - the average reward per step is 2 (it does not vary significantly changing
#    num of episodes and num of steps).
#    Considering that staying in state 3 or 2 gives a reward of 5 per step an
#    an average reward of 2 means that the agent spend more time in the less
#    rewarding states
#
#                    
#interesting discussions on the concepts used here can be found at adventuresinmachinelearning.com

import numpy as np
import matplotlib.pyplot as plt
import random

#class definition of particle object
class obj:
    
    def __init__(self):
        self.X = 5
        self.vx = 0
        
    def update_obj(self):
        self.X = self.X + self.vx*t_step

class particle_sim:
        
    def __init__(self):
        self.X = 0
        self.vx = 0
        if random.random() < 0.5:
            self.X = 10
            self.vx = -1*random.randint(1,2)
        elif random.random() >= 0.5:
            self.X = 0
            self.vx = random.randint(1,2)
        self.state = None
        self.counter = 0
        #X can be initialized to position 10 or 0
        #initial speed is computed consequently to have move towards o
        
    def get_state(self,obj):
        d = self.X - obj.X # distance between X and object o
        if d < 0:
            if d >= -3:
                self.state = 2
            else:
                self.state = 0
        if d > 0:
            if d <= 3:
                self.state = 3
            else:
                self.state = 1
        elif d == 0:
            self.state = 4   

    def step(self,a):
        if a == 0:
            self.vx = -1
        if a == 1:
            self.vx = 1
        self.X = self.X + self.vx*t_step
        self.counter += 1 #increment step count

    def reward(self):
        if self.state == 0 or self.state == 1:
            return 1
        elif self.state == 2 or self.state == 3:
            return 5
        elif self.state == 4:
            return -20

    def end_game(self):
        if self.state == 4:
            return True, 0
        else:
            if self.counter >= game_n:
                return True, 1
            if self.counter < game_n:
                return False, 0
 
############################  

def run_game(table,log):
    #Test - run game function
    p = particle_sim()
    o = obj()
    p.get_state(o)
    tot_reward = 0
    done = False
    while not done:
        s = p.state        
        a = np.argmax(table[s,:])
        p.step(a)
        o.update_obj()
        p.get_state(o)
        r = p.reward()
        tot_reward += r
        end, win = p.end_game()
        if end:
            if log:
                wins_eval.append(win)
            done = True
        
    del p
    del o
    return tot_reward

#q-learning agent
    
def q_learning(num_episodes=500):
    #Q-learning agent
    print("Q-learning agent\n")
    q_table = np.zeros((5,2))
    gamma = 0.95
    learning_rate = 0.8

    for ep in range(num_episodes):
        print("--new episode--",ep+1,"\n")
        p = particle_sim()
        o = obj()
        p.get_state(o)
        tot_reward = 0
        done = False
        while not done:
            s = p.state
            if np.sum(q_table[s,:]) == 0 or random.random() > 0.8:
                a = np.random.randint(0,1)
            else:
                a = np.argmax(q_table[s,:])
            p.step(a)
            o.update_obj()
            p.get_state(o)
            r = p.reward()
            q_table[s,a] += r + learning_rate*(gamma*np.max(q_table[p.state,:]) - q_table[s,a])
            tot_reward += r
            end, win = p.end_game() # verify end game condition
            if end:
                wins_train.append(win)
                done = True
        del p
        del o
        reward_train.append(tot_reward)
    return q_table

#############################

#game_n: maximum number of steps in an episode
#episodes: episodes for training
#episodes_eval: episodes for evaluation

t_step = 0.5
game_n = 20 #
episodes = 100 #
episodes_eval = 50  #

wins_train = []
wins_eval= []
reward_train = []
reward_eval = []

#training phase
table = q_learning(episodes)
print(table)

#evaluation phase
for i in range(0,episodes_eval):
    reward_eval.append(run_game(table,True))


print("win ratio in training: %.2g\n" % (sum(wins_train)/episodes))
print("win ratio in evaluation: %.2g\n" % (sum(wins_eval)/episodes_eval))
print("average reward in evaluation %.2g\n" % (sum(reward_eval)/episodes_eval))
print("average reward per step in evaluation %.2g\n" % (sum(reward_eval)/(episodes_eval*game_n)))


fig = plt.Figure()
plt.subplot(221)
plt.plot(wins_train,label='win during training')
plt.legend(loc=3)

plt.subplot(222)
plt.plot(reward_train,label='rewards during training')
plt.legend(loc=3)

plt.subplot(223)
plt.plot(wins_eval,label='win during evaluation')
plt.legend(loc=3)


plt.subplot(224)
plt.plot(reward_eval,label='rewards during evaluation')
plt.legend(loc=3)

plt.show()

del wins_train,wins_eval,reward_train,reward_eval,table
