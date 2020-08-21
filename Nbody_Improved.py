#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:36:55 2019

@author: christian
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import scipy.constants

print('NBody START')

show_phase=False
show_pos=True
dim = 2
#mass = np.array([1,1,1,1,1,1,1],dtype=np.float64)
mass = np.ones(3,dtype=np.float64)

pos = 2*np.random.rand(len(mass),dim)-1
vel = 1*np.random.rand(len(mass),dim)-0.5
'''Figure-8'''
#pos = np.array([[0.9700436,-0.24308753],[-0.9700436,0.24308753],[0,0]],dtype=np.float64)
#vel = np.array([[0.466203685,0.43236573],[0.466203685,0.43236573],[-2*0.466203685,-2*0.43236573]],dtype=np.float64)

no_objects = len(mass)

rows=[]
for i,row in enumerate(vel):
    rows.append(row*mass[i])
rows = np.array(rows)

#vcentre = np.sum(np.multiply(mass,vel),axis=0)/np.sum(mass)
vcentre=np.sum(rows,axis=0)/np.sum(mass)

#print(np.sum(np.multiply(mass,vel),axis=1))
for obj in vel:
    obj-=vcentre

#translate so that COM at 0,0
com_coords=[]    
for i,elem in enumerate(pos):
    com_coords.append(mass[i]*elem)
com_coords = np.sum(com_coords,axis=0)/np.sum(mass)
for obj in pos:
    obj-=com_coords
    
if show_phase==True:
    pos2 = pos
    vel2=vel
dt=0.01

def dydt(y,timestep=dt):
    soften = dt*10 #Soften the gravitational force dependent on the time-step.
    posit = y[0:len(y)-1:2]
    veloc = y[1:len(y):2]
    
    r_matrix = np.zeros((len(mass),len(mass)),dtype=np.object)
    for i in range(len(mass)):
        for j in range(len(mass)):
            if i > j:
                r_val = posit[i]-posit[j]
                r_matrix[j][i] = r_val/(np.linalg.norm(r_val)**3+np.linalg.norm(r_val)*soften**2)
                r_matrix[i][j] = -r_matrix[j][i]

    deriv = np.zeros((len(mass)*2,dim))
    deriv[0:len(deriv)-1:2] = veloc
    deriv_index = np.arange(1,len(mass)*2,2)
    #print(r_matrix)
    
    mult = np.dot(r_matrix,mass)
    for i,item in enumerate(mult):
        deriv[deriv_index[i]] += mult[i]
#    deriv[1:len(deriv):2] *= scipy.constants.G  #Real-world correction
   
    return deriv


fig = plt.figure()
ax1=fig.add_subplot(1,1,1)   
ax1.set_xlim([-5,5])
ax1.set_ylim([-5,5])

#colour_matrix = ['blue','green','red','orange','magenta','cyan','yellow','brown','red','red','red','red','red','red','red']

bodies = []
lines=[]
linepos=np.zeros((10000,len(mass),dim))

for ind,m in enumerate(mass):
    body, = ax1.plot([pos[ind,0]],[pos[ind,1]],'o',#color=colour_matrix[ind],
                     markersize=3*m,zorder=3)
    bodies.append(body)
    line, = ax1.plot([pos[ind,0]],[pos[ind,1]],#color=colour_matrix[ind],
                     zorder=-1,linewidth=0.25,alpha=0.5)
    lines.append(line)
com, = ax1.plot([0],[0],'*',markersize=5 )
time_text = plt.text(0.02,0.9,'hi',fontsize=14,transform=ax1.transAxes)
fps_text = plt.text(0.9,0.99,'hi',fontsize=12,transform=ax1.transAxes)

prev_time= time.time()
times=[]
sim_time = 0
def animate(k):
    global sim_time
    sim_time+=dt
    global prev_time
    global linepos
    y=[]
    
    for i,elem in enumerate(pos):
        y.append(elem)
        y.append(vel[i])
    y=np.array(y)
#    print('y:',y)
    bef = time.time()
    k1 = dt*dydt(y,dt); #print('k1:',k1)
    k2 = dt*dydt(y+k1/2.0,dt); #print('k2:',k2)
    k3 = dt*dydt(y+k2/2.0,dt); #print('k3:',k3)
    k4 = dt*dydt(y+k3,dt)
    #print('k4:',k4)
    dy = k1/6.0 + k2/3.0 +k3/3.0 + k4/6.0
#    print('RK4 calculations: '+str(time.time()-bef))
    
    newpos = dy[0:len(dy)-1:2]
    newvel = dy[1:len(dy):2]
    
    for i in range(len(newpos)):
        pos[i] += newpos[i]
        vel[i] += newvel[i]
    
    times.append(time.time()-prev_time)
    prev_time = time.time()
    
    bef = time.time()
    min_size=3.5

    bef2=time.time()
    max1 = abs(np.max(pos))
    max2 = abs(np.min(pos))
    max_pos = np.max([max1,max2])+2
#    if (max_pos-3) > min_size-2:
#        ax1.set_xlim([-max_pos,max_pos])
#        ax1.set_ylim([-max_pos,max_pos])
#
#    else:
#        ax1.set_xlim([-min_size,min_size])
#        ax1.set_ylim([-min_size,min_size])
#        
#    print('Axis sizing :'+str(time.time()-bef2))
    
    pos_sum=[]
    
    linepos[k] = pos
    
    if 0 in linepos[-1][0]:
        linepos[np.argmin(abs(linepos),axis=0)[0]] = pos
        lil_linepos=linepos[linepos != 0]
        lil_linepos = lil_linepos.reshape(int(len(lil_linepos)/len(mass)/dim),len(mass),dim)
    else:
        linepos=np.delete(linepos,0,axis=0)
        linepos = np.concatenate((linepos,pos[None,...]),axis=0)
        lil_linepos=np.array(linepos)
    for ind,line in enumerate(lines):
        line.set_data(lil_linepos[:,ind,0],lil_linepos[:,ind,1])
    for ind,body in enumerate(bodies):
        body.set_data(pos[ind,0],pos[ind,1])
        pos_sum.append(mass[ind]*pos[ind,:])
    pos_sum = np.array(pos_sum)
    pos_sum=np.sum(pos_sum,axis=0)/np.sum(mass)
    com.set_data(pos_sum[0],pos_sum[1])
    time_text.set_text("Time: {0:.1f}s".format(sim_time))
    fps_text.set_text("FPS: {0:.1f}".format(1/np.mean(times)))
#    print('Cycle complete.')

if show_phase==True:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    p_coords =[]
    
    for m,v in zip(mass,vel):
        p_coords.append(m*v)
    p_coords = np.array(p_coords,dtype=np.float64)
    
    lines2=[]
    linepos2=np.zeros((10000,len(mass),dim))
    
    bods = []
    for ind,coord in enumerate(p_coords):
        bod, = ax2.plot([coord[0]],[coord[1]],marker='o',#color=colour_matrix[ind],
                        markersize= mass[ind],zorder=3)
        bods.append(bod)
        lin, = ax2.plot([coord[0]],[coord[1]],#color=colour_matrix[ind],
                        zorder=-1,linewidth=0.25,alpha=0.5)
        lines2.append(lin)
        
def animate2(k):
    global sim_time
    sim_time+=dt
    global prev_time
    global linepos2
    y=[]
    
    for i,elem in enumerate(pos2):
        y.append(elem)
        y.append(vel2[i])
    y=np.array(y)
#    print('y:',y)
    bef = time.time()
    k1 = dt*dydt(y,dt); #print('k1:',k1)
    k2 = dt*dydt(y+k1/2.0,dt); #print('k2:',k2)
    k3 = dt*dydt(y+k2/2.0,dt); #print('k3:',k3)
    k4 = dt*dydt(y+k3,dt)
    #print('k4:',k4)
    dy = k1/6.0 + k2/3.0 +k3/3.0 + k4/6.0
#    print('RK4 calculations: '+str(time.time()-bef))
    
    newpos = dy[0:len(dy)-1:2]
    newvel = dy[1:len(dy):2]
    
    for i in range(len(newpos)):
        pos2[i] += newpos[i]
        vel2[i] += newvel[i]
    times.append(time.time()-prev_time)
    prev_time = time.time()
    
    bef = time.time()

    p_coords =[]
    for m,v in zip(mass,vel2):
        p_coords.append(m*v)
    p_coords = np.array(p_coords,dtype=np.float64)
    linepos2[k] = p_coords
    
    if 0 in linepos[-1][0]:
        linepos2[np.argmin(abs(linepos2),axis=0)[0]] = p_coords
        short=linepos2[linepos2 != 0]
        short = short.reshape(int(len(short)/len(mass)/dim),len(mass),dim)
    else:
        linepos2=np.delete(linepos2,0,axis=0)
        linepos2 = np.concatenate((linepos2,p_coords[None,...]),axis=0)
        short=np.array(linepos2)
    for i,line2 in enumerate(lines2):
        line2.set_data(short[:,i,0],short[:,i,1])
    for i,bod in enumerate(bods):
        bod.set_data(p_coords[i,0:2])

    max1 = abs(np.max(short))
    max2 = abs(np.min(short))
    max_pos = np.max([max1,max2])+2
    min_size=3.5
    
    if (max_pos-2) > min_size-2 and False:
        ax2.set_xlim([-max_pos,max_pos])
        ax2.set_ylim([-max_pos,max_pos])

    else:
        ax2.set_xlim([-min_size,min_size])
        ax2.set_ylim([-min_size,min_size])
        
if show_phase==True:
    ani2=animation.FuncAnimation(fig2,animate2,interval=0)
if show_pos==True:
    ani=animation.FuncAnimation(fig,animate,interval=0)