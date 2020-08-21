#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:25:46 2019

@author: christian
"""

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
import random
import astropy.coordinates
from astropy.constants import au
import astropy.time
from astropy import units as u
import datetime
from NBody_dydt import dydt
import sys

print('NBody START')
dim = 3
use_ss = True
use_current_time = True

phase_space=False
#pov_earth=True

#Lowercase only. For barycentric, leave None
pov_name = None


if use_ss==True:
    if use_current_time:
        now=astropy.time.Time(time.time(),format='unix')
    else:
        now = astropy.time.Time('2019-01-01T00:00:00',format='isot',scale='utc')
    now_datetime = now.to_datetime()
    #get current solar system data
    pos = []
    vel = []
    mass=[]
    body_names = astropy.coordinates.solar_system_ephemeris.bodies
    mass_dict = {
            "sun":1988500000e21,
            "jupiter":1898200e21,
            "saturn":568340e21,
            "uranus":86813e21,
            "neptune":102413e21,
            "earth":5972.4e21,
            "venus":4867.5e21,
            "mars":641.7e21,
            "mercury":330.1e21
            }
    
    colour_dict = {
            "sun":'darkorange',
            "mercury":'grey',
            "venus":'magenta',
            "earth":'royalblue',
            "mars":'red',
            "jupiter":'peru',
            "saturn":'gold',
            "uranus":'turquoise',
            "neptune":'blue'}

    new_body = '2I/Borisov'
    colour_dict[new_body] = 'green'
#    body_names+=(new_body,)
    mass_dict[new_body] = 1e13
    body_names = [x for x in body_names if 'moon' not in x]
#    body_names.remove(')
    for i,body in enumerate(body_names):
        if 'moon' not in body and new_body not in body:
            cart = astropy.coordinates.get_body_barycentric_posvel(body,now)
            converted_pos = cart[0].get_xyz().to(u.m).value
            converted_vel = cart[1].get_xyz().to(u.m/u.s).value
            
            pos.append(converted_pos)
            vel.append(converted_vel)
            mass.append(mass_dict[body])
        elif new_body in body:
            pos_not=[4.224906810,5.761187925,12.744462546]
            vel_not=[-0.008344761 , -0.005496505 , -0.016981821]
            
            pos_not = [x*au.value for x in pos_not]
            vel_not = [x*au.value/86400 for x in vel_not]
            pos.append(pos_not)
            vel.append(vel_not)
            
            mass.append(mass_dict[body])
        
    pos=np.array(pos,dtype=np.float64)
#    print('pos:',pos)
    vel=np.array(vel,dtype=np.float64)
#    print('vel:',vel)
    mass = np.array(mass,dtype=np.float64)
    
    def marker_size_rule(m):
        return np.log(m/np.min(mass))/np.log(200)+1#(1-100/(100+np.log10(m)))
    
else:
    mass = np.array([8,8,8,8,8,8,8,8,8],dtype=np.float64)
    pos = 15*np.random.rand(len(mass),dim)-7.5
    vel = 10*np.random.rand(len(mass),dim)-7.5

    def marker_size_rule(m):
        return m

#Get body-POV
if pov_name is None:
    pov_body = None
elif pov_name is not None:
    if pov_name in body_names:
        pov_body = body_names.index(pov_name)
    else:
        print('Please input one of the following bodies: ',body_names)
        sys.exit()

no_objects = len(mass)

rows=[]
for i,row in enumerate(vel):
    rows.append(row*mass[i])
rows = np.array(rows)

vcentre=np.sum(rows,axis=0)/np.sum(mass)

for obj in vel:
    obj-=vcentre
    
#translate so that COM at 0,0
com_coords=[]    
for i,elem in enumerate(pos):
    com_coords.append(mass[i]*elem)
com_coords = np.sum(com_coords,axis=0)/np.sum(mass)
for obj in pos:
    obj-=com_coords

if pov_body is not None:
    pos_pov = pos-pos[pov_body]
    vel_pov = vel-vel[pov_body]
    
dt=86400*0.5

fig = plt.figure(figsize=(10,10))
ax1=fig.add_subplot(111,projection='3d')  

bodies = []
lines=[]

if pov_body is None:
    linepos=np.zeros((100,len(mass),dim))
    linepos[0] = pos
elif pov_body is not None:
    linepos=np.zeros((1,len(mass),dim))
    linepos[0] = pos_pov
if phase_space:
    linepos2=np.zeros((100,len(mass),dim))
    p_coords =[]
    
    if pov_body is None:
        for m,v in zip(mass,vel):
            p_coords.append(v)
    elif pov_body is not None:
         for m,v in zip(mass,vel_pov):
            p_coords.append(v)
       
    p_coords = np.array(p_coords,dtype=np.float64)
    
    lines2=[]
    
    for ind,m in enumerate(mass):
        body, = ax1.plot([p_coords[ind,0]],[p_coords[ind,1]],[p_coords[ind,2]],label=body_names[ind],marker='o',
                         color=colour_dict[body_names[ind]],markersize=marker_size_rule(m),linestyle='None',zorder=-1)
        bodies.append(body)
        line, = ax1.plot([p_coords[ind,0]],[p_coords[ind,1]],[p_coords[ind,2]],
                         label='',color=colour_dict[body_names[ind]],linewidth=0.4,alpha=0.9,zorder=3)
        lines.append(line)


if not phase_space:
    if pov_body is None:
        for ind,m in enumerate(mass):
            body, = ax1.plot([pos[ind,0]],[pos[ind,1]],[pos[ind,2]],label=body_names[ind],marker='o',
                             color=colour_dict[body_names[ind]],markersize=marker_size_rule(m),linestyle='None',zorder=-1)
            bodies.append(body)
            line, = ax1.plot([pos[ind,0]],[pos[ind,1]],[pos[ind,2]],
                             label='',color=colour_dict[body_names[ind]],linewidth=0.4,alpha=0.9,zorder=3)
            lines.append(line)
    elif pov_body is not None:
        for ind,m in enumerate(mass):
            body, = ax1.plot([pos_pov[ind,0]],[pos_pov[ind,1]],[pos_pov[ind,2]],label=body_names[ind],marker='o',
                             color=colour_dict[body_names[ind]],markersize=marker_size_rule(m),linestyle='None',zorder=-1)
            bodies.append(body)
            line, = ax1.plot([pos_pov[ind,0]],[pos_pov[ind,1]],[pos_pov[ind,2]],
                             label='',color=colour_dict[body_names[ind]],linewidth=0.4,alpha=0.9,zorder=3)
            lines.append(line)        
    
if use_ss == False:
    com, = ax1.plot([0],[0],'*',markersize=5 )

time_text = ax1.text2D(0.02,0.9,'hi',fontsize=14,transform=ax1.transAxes)
fps_text = ax1.text2D(0.02,0.85,'hi',fontsize=14,transform=ax1.transAxes)
prev_time= time.time()
times=[]
sim_time = 0

ax1.set_facecolor((0.9,0.9,0.9))
ax1.legend(loc='upper right')
ax1.view_init(elev=90,azim=45)

#Axis Limits
if use_ss == True:
    if not phase_space:
        max1 = abs(np.max(pos[:-3]))
        max2 = abs(np.min(pos[:-3]))
        zmax1=abs(np.max(pos[:,2]))
        zmax2=abs(np.min(pos[:,2]))
        max_pos = np.max([max1,max2])*1.15/au.value
        zmax_pos=np.max([zmax1,zmax2])*1.15/au.value
        ax1.set_xlim3d([-max_pos,max_pos])
        ax1.set_ylim3d([-max_pos,max_pos])
        ax1.set_zlim3d([-max_pos,max_pos])
    elif phase_space:
        max1 = abs(np.max(p_coords[:-3]))
        max2 = abs(np.min(p_coords[:-3]))
        zmax1=abs(np.max(p_coords[:,2]))
        zmax2=abs(np.min(p_coords[:,2]))
        max_pos = np.max([max1,max2])*1.15
        zmax_pos=np.max([zmax1,zmax2])*1.15
        ax1.set_xlim3d([-max_pos,max_pos])
        ax1.set_ylim3d([-max_pos,max_pos])
        ax1.set_zlim3d([-max_pos,max_pos])


def animate(k):
    global linepos
    global sim_time
    sim_time+=dt
    global prev_time
    y=[]
    for i,elem in enumerate(pos):
        y.append(elem)
        y.append(vel[i])
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
        pos[i] += newpos[i]
        vel[i] += newvel[i]
    
    if not phase_space:
            if pov_body is None:
                if 0 in linepos[-1][0]:
                    linepos[np.argmin(abs(linepos),axis=0)[0]] = pos
                    lil_linepos=linepos[linepos != 0]
                    lil_linepos = lil_linepos.reshape(int(len(lil_linepos)/len(mass)/dim),len(mass),dim)
                else:
        #            linepos=np.delete(linepos,0,axis=0)
                    linepos = np.concatenate((linepos,pos[None,...]),axis=0)
                    lil_linepos=np.array(linepos)
            elif pov_body is not None:
                pos_pov = pos - pos[pov_body]
                linepos = np.concatenate((linepos,pos_pov[None,...]),axis=0)
                lil_linepos=np.array(linepos)

    elif phase_space:
        global linepos2
        p_coords =[]
        for m,v in zip(mass,vel):
            p_coords.append(v)
        p_coords = np.array(p_coords,dtype=np.float64)
        if 0 in linepos[-1][0]:
            linepos2[np.argmin(abs(linepos2),axis=0)[0]] = p_coords
            lil_linepos=linepos2[linepos2 != 0]
            lil_linepos = lil_linepos.reshape(int(len(lil_linepos)/len(mass)/dim),len(mass),dim)
        else:
            linepos2=np.delete(linepos2,0,axis=0)
            linepos2 = np.concatenate((linepos2,p_coords[None,...]),axis=0)
            lil_linepos=np.array(linepos2)

    
    times.append(time.time()-prev_time)
    prev_time = time.time()
    
#    bef = time.time()

#    if use_ss == False:
#        max1 = abs(np.max(pos))
#        max2 = abs(np.min(pos))
#        max_pos = np.max([max1,max2])+3
    #    if (max_pos-3) > min_size-3:
    #        ax1.set_xlim3d([-max_pos,max_pos])
    #        ax1.set_ylim3d([-max_pos,max_pos])
    #        ax1.set_zlim3d([-max_pos,max_pos])
    #
    #    else:
    #        ax1.set_xlim3d([-min_size,min_size])
    #        ax1.set_ylim3d([-min_size,min_size])
    #        ax1.set_zlim3d([-min_size,min_size])
        
    if use_ss == True:
        if phase_space == False:
            if pov_body is None:
                for ind,body in enumerate(bodies):
                    body.set_data(pos[ind,0:2]/au.value)
                    body.set_3d_properties(pos[ind,2]/au.value)
                    lines[ind].set_data(lil_linepos[:,ind,0]/au.value,lil_linepos[:,ind,1]/au.value)
                    lines[ind].set_3d_properties(lil_linepos[:,ind,2]/au.value)
            elif pov_body is not None:
                for ind,body in enumerate(bodies):
                    body.set_data(pos_pov[ind,0:2]/au.value)
                    body.set_3d_properties(pos_pov[ind,2]/au.value)
                    lines[ind].set_data(lil_linepos[:,ind,0]/au.value,lil_linepos[:,ind,1]/au.value)
                    lines[ind].set_3d_properties(lil_linepos[:,ind,2]/au.value)

        elif phase_space == True:
            for ind,body in enumerate(bodies): 
                body.set_data(p_coords[ind,0:2])
                body.set_3d_properties(p_coords[ind,2])
                lines[ind].set_data(lil_linepos[:,ind,0],lil_linepos[:,ind,1])
                lines[ind].set_3d_properties(lil_linepos[:,ind,2])

    
    if use_ss==False:
        pos_sum=[]
        for ind,body in enumerate(bodies):
#            body.set_data(pos[ind,0],pos[ind,1])
            body.set_data(pos[ind,0:2])
            body.set_3d_properties(pos[ind,2])
            pos_sum.append(mass[ind]*pos[ind,:])
        pos_sum = np.array(pos_sum)
        pos_sum=np.sum(pos_sum,axis=0)/np.sum(mass)
#        print(pos_sum)

        com.set_data(pos_sum[0],pos_sum[1])
#    print(k)
    simul_date = now_datetime + datetime.timedelta(seconds=sim_time)
    time_text.set_text(str(simul_date)) #\n{0:.1f}d".format(sim_time/86400))
    fps_text.set_text("FPS: {0:.1f}".format(1/np.mean(times[-5:])))

#    print('Plotting time: '+str(time.time()-bef))
#    print('Cycle complete.')
    
ani=animation.FuncAnimation(fig,animate,interval=10)
#ani.save('./animation.gif', writer='imagemagick', fps=40)


