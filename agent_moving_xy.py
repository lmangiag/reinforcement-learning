#an agent is trained to be able to move control the movements of a simple particle 
#and keep a safety distance from a target point T
#
#interesting discussions on the concepts used here can be found at adventuresinmachinelearning.com

import numpy as np
import matplotlib.pyplot as plt
import random
import math

#class definition of particle object
class particle_sim:

    def __init__(self):
        self.X = 0
        self.Y = 0
        self.vx = 1*random.random()
        self.vy = 1*random.random()
        self.Tx = 2*random.random()
        self.Ty = 2*random.random()
        self.state = 0

    def get_state(self):
        R = (self.X - self.Tx)*(self.X - self.Tx) + (self.Y - self.Ty)*(self.Y - self.Ty)
        if R >= 1 and R < 2:
            self.state = 0
        if R >= 2 :
            self.state = 1
        if R < 1.5 and R > 0.1:
            self.state = 2
        if R <= 0.1 :            
            self.state = 3

    def step(self,a):
        if a == 0:
            self.vx = self.vx
        if a == 1:
            self.vx = self.vx -1
        if a == 2:
            self.vx = self.vx +1
        if a == 3:
            self.vy = self.vy
        if a == 4:
            self.vy = self.vy -1
        if a == 5:
            self.vy = self.vy +1
        self.X = self.X + self.vx*t_step
        self.Y = self.Y + self.vy*t_step


    def reward(self):
        if self.state == 0:
            return 1
        if self.state == 1:
            return 5
        if self.state == 2:
            return 2
        if self.state == 3:
            return -10

    def end_game(self):
        if self.state == 3:
            return True
        else:
            return False
 
############################

def run_game(table):
    #Test - run game function
    p = particle_sim()
    p.get_state()
    s = p.state
    tot_reward = 0
    done = False
    counter = 0
    while not done:
        s = p.state        
        a = np.argmax(table[s,:])
        p.step(a)
        p.get_state()
        r = p.reward()
        done = p.end_game()
        tot_reward += r
        counter += 1
        if counter > 10:
            done = True
    del p
    return tot_reward

#a simple agent based on maximization of immediate reward

def sum_reward_agent(wins,num_episodes=2):
    #Learn - naive sum reward agent
    print("Sum reward agent\n")
    r_table = np.zeros((4,3))

    for ep in range(num_episodes):
        print("--new episode--",ep,"\n")
        p = particle_sim()
        p.get_state()
        s = p.state
        done = False
        counter = 0
        while not done:
            s = p.state
            if np.sum(r_table[s,:]) == 0 or random.random() > 0.9:
                a = np.random.randint(0,2)
            else:
                a = np.argmax(r_table[s,:])
            p.step(a)
            #print("s:",s,"a:",a,"\n")
            p.get_state()
            new_s = p.state            
            r = p.reward()
            done = p.end_game()
            r_table[s,a] += r
            counter += 1
            if p.end_game():
                wins.append(0)
                done = True
            elif counter > 10:
                wins.append(1)
                done = True
        del p
    return r_table

#an agent based on Q-learning that consider future reward with a discount factor

def q_learning(wins,num_episodes=2):
    #Q-learning agent
    print("Q-learning agent\n")
    q_table = np.zeros((4,6))
    gamma = 0.95
    learning_rate = 0.8

    for ep in range(num_episodes):
        print("--new episode--[%i]\n" % ep)
        p = particle_sim()
        p.get_state()
        s = p.state
        done = False
        counter = 0
        while not done:
            s = p.state
            if np.sum(q_table[s,:]) == 0:
                a = np.random.randint(0,2)
            else:
                a = np.argmax(q_table[s,:])
            p.step(a)
            #print("s:",s,"a:",a,"\n")
            p.get_state()
            new_s = p.state            
            r = p.reward()
            done = p.end_game()
            q_table[s,a] += r + learning_rate*(gamma*np.max(q_table[new_s,:]) - q_table[s,a])
            counter += 1
            if p.end_game():
                wins.append(0)
                done = True
            elif counter > 10:
                wins.append(1)
                done = True
        del p
    return q_table


#############################

t_step = 0.5
episodes = 1000

wins = []
rewards_eval = []

#training phase
#m_table = sum_reward_agent(wins,episodes)
m_table = q_learning(wins,episodes)
#evaluation phase
for i in range(0,100):
    rewards_eval.append(run_game(m_table))
    

print("Wins in training: %f\n" % (100*sum(wins)/episodes))

plt.subplot(211)
plt.plot(wins,label='win during training')
plt.legend(loc=3)

plt.subplot(212)
plt.plot(rewards_eval,label='rewards during evaluation')
plt.legend(loc=3)

plt.show()

#############################




