#Particles
# simulate behaviour of simple particles in empty space

import random
import matplotlib.pyplot as plt
import numpy as np
import math

#Boundaries for particle instantiation

bound_XA = 0
bound_XB = 6
bound_YA = 0
bound_YB = 6


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
        if self.near:
            self.vX = self.vX/2
            self.vY = self.vY/2
   
        
##           #
## Functions #
##           #

# update positions for plot
def update_pos(particles_pos,particles_list,particles_num):
    for i in range(0,particles_num):
        particles_pos[i][0] = particles_list[i].X
        particles_pos[i][1] = particles_list[i].Y

# move particles
def move_particles(particles_pos,particles_list,particles_num,t_step):
    for i in range(0,particles_num-1):
        particles_list[i].move()

## Functions for interaction calculation#
                                      

#check if particle1 collides with particle2
def if_collide(particle1,particle2,r1,r2):
    R = r1 + r2
    if abs(particle1.X - particle2.X) < R and abs(particle1.Y - particle2.Y) < R:
        return True
    else:
        return False

def if_near(particle1,particle2):
    N = 1
    if abs(particle1.X - particle2.X) < N and abs(particle1.Y - particle2.Y) < N:
        if abs(particle1.X - particle2.X) > N-0.1 and abs(particle1.Y - particle2.Y) > N-0.1:
            return True
    else:
        return False

#extract lists with incidexes of colliding particles
def particles_collision(particles_list):
    collisions = []
    for i in range(0,particles_num - 1):
        for k in range(i+1,particles_num - 1):
            #if if_collide(particles_list[i],particles_list[k],particles_list[i].r,particles_list[k].r):
                 #collisions.append((i,k))
                 #print(collisions)
                 #particles_list[i].vX = -1*particles_list[i].vX 
                 #particles_list[i].vY = -1*particles_list[i].vY
                 #particles_list[k].vX = -1*particles_list[k].vX 
                 #particles_list[k].vY = -1*particles_list[k].vY
            if if_near(particles_list[i],particles_list[k]):
                particles_list[i].near = True
                particles_list[k].near = True
    return collisions


## main


#Initialize Physical parameters
particles_num = 50
maxX = bound_XB
maxY = bound_YB
maxvX = 2
maxvY = 2
mass = 1
radius = 0.1

#Initialize simulation
t_step = 0.1
iterations=100

particles_list = {}
particles_pos = np.zeros((particles_num,2))

for particle_idx in range(0,particles_num):
    particles_list[particle_idx] = Particle(random.random()*maxX, random.random()*maxY, random.random()*maxvX - maxvX/2, random.random()*maxvY - maxvY/2, mass, t_step, radius)

update_pos(particles_pos,particles_list,particles_num)

#Draw the figure
plt.ion()
fig, ax = plt.subplots()
sc = ax.scatter(particles_pos[:,0],particles_pos[:,1])
plt.xlim(bound_XA,bound_XB+1)
plt.ylim(bound_YA,bound_YB+1)

plt.draw()

for i in range(iterations):
    move_particles(particles_pos, particles_list, particles_num, t_step)
    particles_collision(particles_list)
    #update_particles_v(particles_list)

    update_pos(particles_pos,particles_list,particles_num)
    sc.set_offsets(np.c_[particles_pos[:,0],particles_pos[:,1]])
    fig.canvas.draw_idle()
    plt.pause(t_step)
    
plt.waitforbuttonpress()


