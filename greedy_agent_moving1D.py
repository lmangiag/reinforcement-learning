#an agent is trained with a reward-based algorithm to be able to control 
#a simple particle in 1D and avoid collision with an object o
#
#the state is computed calculating relative distance between particle X and o
#s = 2: collision -> endgame reward -10
#s = 1: relative distance d > 0 -> reward +1
#s = 0: relative distance d < 0 -> reward +1
#
#the agent can choose to perform 2 actions
#a = 0: vx = -1
#a = 1: vx = +1
#
#a greedy agent, based on maximization of instantaneous reward is able to learn
#a policy that performs well with fixed object: change vx to increase distance
#from o
#
#changing the actions allowing decrease or increase of vx by 1 gives a
#different result: the agent learns to put vx to 0 to avoid collision
#
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
            self.state = 0
        elif d > 0:
            self.state = 1
        elif d == 0:
            self.state = 2   

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
        elif self.state == 2:
            return -10

    def end_game(self):
        if self.state == 2:
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

#a simple agent based on maximization of immediate reward

def greedy_agent(num_episodes=2):
    #Learn - naive sum reward agent
    print("Sum reward agent\n")
    r_table = np.zeros((3,2))

    for ep in range(num_episodes):
        print("--new episode--",ep+1,"\n")
        p = particle_sim()
        o = obj()
        p.get_state(o)
        tot_reward = 0
        done = False
        while not done:
            s = p.state
            if np.sum(r_table[s,:]) == 0:
                a = np.random.randint(0,1)
            else:
                a = np.argmax(r_table[s,:])
            p.step(a)
            o.update_obj()
            p.get_state(o)
            r = p.reward()
            r_table[s,a] += r
            tot_reward += r
            end, win = p.end_game() # verify end game condition
            if end:
                wins_train.append(win)
                done = True
        del p
        del o
        reward_train.append(tot_reward)
    return r_table


#############################

t_step = 0.5
game_n = 10 #maximum number of steps in an episode
episodes = 100 #episodes for training
episodes_eval = 10  #episodes for evaluation

wins_train = []
wins_eval= []
reward_train = []
reward_eval = []
X_pos = []

#training phase
r_table = greedy_agent(episodes)
print(r_table)

#evaluation phase
for i in range(0,episodes_eval):
    reward_eval.append(run_game(r_table,True))

print("win ratio in training: %.2g\n" % (sum(wins_train)/episodes))
print("win ratio in evaluation: %.2g\n" % (sum(wins_eval)/episodes_eval))

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
plt.legend(loc=3)

plt.subplot(224)
plt.plot(reward_eval,label='rewards during evaluation')
plt.legend()

plt.show()



