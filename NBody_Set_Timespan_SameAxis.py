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
import matplotlib
matplotlib.use('Agg')
print('NBody START')

matplotlib.rc('xtick', labelsize=21)
matplotlib.rc('ytick', labelsize=21)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Christian CB'), bitrate=1800)


show_phase=True
show_pos=True
dim = 2

duration=15 #duration in seconds
precision  = 0.00005 #precision in seconds

duration+=precision

num_cycles = int(duration/precision)-1

#mass = np.array([1,1,1,1,1,1,1],dtype=np.float64)
mass = np.ones(4,dtype=np.float64)

#np.random.seed(0)
#pos = 2*np.random.rand(len(mass),dim)-1
#np.random.seed(0)
#vel = 1*np.random.rand(len(mass),dim)-0.5

# pos[0,0]+=0.05
# pos[0,1]-=0.05

# '''Figure-8'''
# pos = np.array([[0.9700436,-0.24308753],[-0.9700436,0.24308753],[0,0]],dtype=np.float64)
# vel = np.array([[0.466203685,0.43236573],[0.466203685,0.43236573],[-2*0.466203685,-2*0.43236573]],dtype=np.float64)

'''FOUR QUADRANTS'''
pos = np.array([[0,1],[1,0],[0,-1],[-1,0]],dtype=np.float64)
vel = np.array([[1,0],[0,-1],[-1,0],[0,1]],dtype=np.float64)

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
    
#We want to do a set number of cycles, so initialize the big arrays now.
pos = np.concatenate((pos[None,...],np.zeros((num_cycles,len(mass),dim),dtype=np.float64)),axis=0)
vel = np.concatenate((vel[None,...],np.zeros((num_cycles,len(mass),dim),dtype=np.float64)),axis=0)


if show_phase==True:
    pos2 = pos
    vel2=vel
dt=precision

def dydt(y,mass,dim,timestep=dt):
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

sim_times=[]
sim_time=0
start_time=time.time()
for cyc_i in range(num_cycles):
    if cyc_i==0:
        now=time.time()
    global linepos
    
    sim_time+=dt
    sim_times.append(sim_time)
    global prev_time
    
    y=[]
    for i,elem in enumerate(pos[cyc_i]):
        y.append(elem)
        y.append(vel[cyc_i][i])
    y=np.array(y)
    bef = time.time()
    k1 = dt*dydt(y,mass,dim); #print('k1:',k1)
    k2 = dt*dydt(y+k1/2.0,mass,dim); #print('k2:',k2)
    k3 = dt*dydt(y+k2/2.0,mass,dim); #print('k3:',k3)
    k4 = dt*dydt(y+k3,mass,dim)
    #print('k4:',k4)
    dy = k1/6.0 + k2/3.0 +k3/3.0 + k4/6.0
#    print('RK4 calculations: '+str(time.time()-bef))
    
    newpos = dy[0:len(dy)-1:2]
    newvel = dy[1:len(dy):2]
    
    for i in range(len(newpos)):
        pos[cyc_i+1,i] = pos[cyc_i,i] + newpos[i]
        vel[cyc_i+1,i] = vel[cyc_i,i] + newvel[i]
    if cyc_i==0:
        looptime = time.time()-now
        print('Time per loop: '+str(looptime))
        print('Estimated duration: '+str(looptime*num_cycles)+'s for '+str(num_cycles)+' cycles.')
print('Model complete in: '+str(time.time()-start_time))


#fig = plt.figure(figsize=(6,6))
fig, axs = plt.subplots(1,2,sharey=True,figsize=(24,12))
plt.subplots_adjust(wspace=0.05,left=0.05,right=0.975)
axs[0].set_xlim([-1.5,1.5])
axs[0].set_ylim([-1.5,1.5])
axs[0].grid('on')
axs[0].set_title('Real Space',fontsize=28)

colour_matrix = ['blue','green','red','orange','magenta','cyan','yellow','brown','red','red','red','red','red','red','red']
#colour_matrix = ['blue','blue','blue']
bodies = []
lines=[]

for ind,m in enumerate(mass):
    body, = axs[0].plot([pos[0,ind,0]],[pos[0,ind,1]],'o',color=colour_matrix[ind],
                     markersize=12*m,zorder=3)
    bodies.append(body)
    line, = axs[0].plot([pos[0,ind,0]],[pos[0,ind,1]],color=colour_matrix[ind],
                     zorder=-1,linewidth=3,alpha=0.5)
    lines.append(line)
com, = axs[0].plot([0],[0],'r+',markersize=20 )
time_text = plt.text(0.02,0.89,'hi',fontsize=32,transform=axs[0].transAxes)
#fps_text = plt.text(0.02,0.85,'hi',fontsize=12,transform=ax1.transAxes)

prev_time= time.time()
times=[]
sim_time = 0

def animate_timedur(k):
    global sim_time
    sim_time= sim_time + (0.01/precision)*dt
    global prev_time
    global pos
    global vel
    
    if k % 50 == 0:
        print(k)
    
    times.append(time.time()-prev_time)
    prev_time = time.time()
       
    for ind,body in enumerate(bodies):
        body.set_data(pos[k,ind,0],pos[k,ind,1])
        # if k <= 100:
        lines[ind].set_data(pos[:k,ind,0],pos[:k,ind,1])
        # elif k > 100:
            # lines[ind].set_data(pos[k-100:k,ind,0],pos[k-100:k,ind,1])
            
    for i,line2 in enumerate(lines2):
        line2.set_data(vel[:k,i,0],vel[:k,i,1])
    for i,bod in enumerate(bods):
        bod.set_data(vel[k,i,0],vel[k,i,1])
        
    com_coords=[]    
    for i,elem in enumerate(pos[k,:,:]):
        com_coords.append(mass[i]*elem)
    com_coords = np.sum(com_coords,axis=0)/np.sum(mass)
    com.set_data(com_coords[0],com_coords[1])

    time_text.set_text("Time: {0:.2f}s".format(sim_time)) #\n{0:.1f}d".format(sim_time/86400))
#    fps_text.set_text("FPS: {0:.1f}".format(1/np.mean(times[-5:])))


if show_phase==True:
    #fig2 = plt.figure(figsize=(12,8))

    axs[1].grid('on')
    #time_text2 = plt.text(0.02,0.89,'hi',fontsize=32,transform=ax2.transAxes)
#    fps_text2 = plt.text(0.02,0.85,'hi',fontsize=12,transform=ax1.transAxes)
    
    lines2=[]    
    bods = []
    for ind,coord in enumerate(vel[0]):
        bod, = axs[1].plot([coord[0]],[coord[1]],marker='o',color=colour_matrix[ind],
                        markersize= 12*mass[ind],zorder=3)
        bods.append(bod)
        lin, = axs[1].plot([coord[0]],[coord[1]],color=colour_matrix[ind],
                        zorder=-1,linewidth=3,alpha=0.5)
        lines2.append(lin)
    axs[1].set_xlim([-1.5,1.5])
    axs[1].set_ylim([-1.5,1.5]) 
    axs[1].set_title('Momentum Space',fontsize=28)
if show_pos==True:
    ani=animation.FuncAnimation(fig,animate_timedur,frames=range(0,int(duration/precision-1),int(0.01/precision)),interval=15)
    ani.save('sametest2.mp4',writer=writer)