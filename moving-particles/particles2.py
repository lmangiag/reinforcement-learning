#Particles
# simulate the behaviour of simple particles in empty space with different policies

import random
import matplotlib.pyplot as plt
import numpy as np
import math

# Particle Class declaration #

class Particle:

    def __init__(self, X, Y, vX, vY, mass, t_step, r, near=False):
        self.X = X
        self.Y = Y
        self.vX = vX
        self.vY = vY
        self.mass = mass
        self.r = r
        self.near = near

    def outX(self):
        return self.X

    def outY(self):
        return self.Y

    def outvX(self):
        return self.vX

    def outvY(self):
        return self.vY

    def move(self):
        self.X = self.X + (self.vX * t_step)
        self.Y = self.Y + (self.vY * t_step)
        
## Functions #

def update_pos(particles_pos,particles_dict,particles_num):
    # update positions for plot
    for i in range(0,particles_num):
        particles_pos[i][0] = particles_dict[i].X
        particles_pos[i][1] = particles_dict[i].Y

def move_particles(particles_pos,particles_dict,particles_num,t_step):
    # move particles
    for i in range(0,particles_num-1):
        particles_dict[i].move()

def if_collide(particle1,particle2,r1,r2):
    #check if particle1 collides with particle2
    R = r1 + r2
    if abs(particle1.X - particle2.X) < R and abs(particle1.Y - particle2.Y) < R:
        return True
    else:
        return False

def if_near(particle1,particle2):
    N = 2
    if abs(particle1.X - particle2.X) < N and abs(particle1.Y - particle2.Y) < N:
        return True
    else:
        return False


def particles_collision(particles_dict):
    #extract tuples with indices of colliding particles
    collisions = []
    for i in range(0,particles_num - 1):
        for k in range(i+1,particles_num - 1):
            if if_near(particles_dict[i],particles_dict[k]):
                particles_dict[i].near = True
                particles_dict[k].near = True
    return collisions


#Boundaries for particle instantiation

bound_XA = 0
bound_XB = 8
bound_YA = 0
bound_YB = 8

#Initialization of Physical parameters

particles_num = 50
maxX = bound_XB
maxY = bound_YB
maxvX = 2
maxvY = 2
mass = 1
radius = 0.1

#Initialization of simulation parameters

t_step = 0.1
iterations=100

particles_dict = {} # particles are stored in a dictionary to retrieve each one efficiently
particles_pos = np.zeros((particles_num-1,2))

for particle_idx in range(0,particles_num-1):
    particles_dict[particle_idx] = Particle(random.random()*maxX, random.random()*maxY, random.random()*maxvX - maxvX/2, random.random()*maxvY - maxvY/2, mass, t_step, radius)

#update_pos(particles_pos,particles_dict,particles_num)

#Draw the figure
plt.ion()

#sc = ax.scatter(particles_pos[:,0],particles_pos[:,1])
for key in particles_dict:
    plt.scatter(particles_dict[key].X,particles_dict[key].Y)
plt.xlim(bound_XA,bound_XB+1)
plt.ylim(bound_YA,bound_YB+1)

#plt.draw()

for i in range(iterations):
    move_particles(particles_pos, particles_dict, particles_num, t_step)
    particles_collision(particles_dict)
    plt.pause(t_step)
    
for key in particles_dict:    
    plt.scatter(particles_dict[key].X,particles_dict[key].Y)
reward = 0
for i in range(0,particles_num - 1):
    if particles_dict[i].near:
        reward = reward +1

print("iterations: %i \nnum of particles %i \nreward %i" % (iterations,particles_num,reward))
plt.waitforbuttonpress()


