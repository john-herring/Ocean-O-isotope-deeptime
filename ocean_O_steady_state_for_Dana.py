#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 16:11:25 2018

@author: benjohnson
"""
import numpy as np
import scipy as sp
import scipy.misc
from scipy.integrate import odeint
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
#%% Seawater oxygen isotope exchange model
t = 4.4 #time in Gyr

Wo = 7 #original seawater d18O
W_SS = -1 #steady state 
num_steps = 100 #num of initial model steps
time = np.linspace(0,4.5,num=num_steps) #sample every 250 myr
weath_time_on_twostep = 4.5-2.8#in Ga
weath_time_on = 4.5-2.7
weath_time_early = 4.5-4.43
weath_time_late = 4.5-0.9

# rate constants in Gyr-1, from Muehlenbachs, 1998
k_weath = 8 #nominal 8 continental weathering
k_growth = 1.2 #nominal 1.2 continental growth
k_hiT = 14.6 #nominal 14.6high temperature seafloor 
k_loT = 1.7 #nominal 1.7low temp seafloor/seafloor weathering 
k_W_recycling = 0.6 #nominal 0.6water recycling at subduction zones

#fractionations (permil) btwn rock and water, from Muehlenbachs, 1998 except weathering, which we tuned to reproduce -1permil ocean

Delt_weath =  13  #mueh = 9.6 nominal 13, newer 17
Delt_growth =  9.8 #mueh = 9.8 
Delt_hiT_mid =  1.5 #meuh = 4.1
Delt_hiT = 4.1 
#Delt_hiT_mid = np.zeros(len(time))

Delt_lowT =  9.3  #mueh = 9.3 
Delt_water_recycling = 2.5 # mueh = 2.5 
    

#calculate steady state in 250 myr increments 
del_graniteo = np.linspace(7.8,7.8,num=num_steps)

del_basalto = 5.5
del_WR = 7
bb= 0.1


Delt_hiT_change = (Delt_hiT-Delt_hiT_mid)+(Delt_hiT-Delt_hiT_mid)*0.5*(1+np.tanh((np.subtract(time,weath_time_on)/bb))) -1
Delt_hiT_change_late = (Delt_hiT-Delt_hiT_mid)+(Delt_hiT-Delt_hiT_mid)*0.5*(1+np.tanh((np.subtract(time,weath_time_late)/bb))) -1
Delt_hiT_twostep = (Delt_hiT-Delt_hiT_mid)+(Delt_hiT-Delt_hiT_mid)*0.5*(1+np.tanh((np.subtract(time,weath_time_on_twostep)/bb))) -1

k_growth_change = 0.5*k_growth*(1+np.tanh((np.subtract(time,weath_time_on)/bb)))
k_growth_late = 0.5*k_growth*(1+np.tanh((np.subtract(time,weath_time_late)/bb)))
k_growth_early = 0.5*k_growth*(1+np.tanh((np.subtract(time,weath_time_early)/bb)))



k_weathering_change = 0.5*k_weath*(1+np.tanh((np.subtract(time,weath_time_on)/bb)))
k_weathering_late = 0.5*k_weath*(1+np.tanh((np.subtract(time,weath_time_late)/bb)))
k_weathering_early = 0.5*k_weath*(1+np.tanh((np.subtract(time,weath_time_early)/bb)))

two_step_time = 4.5-2
k_growth_twostep = 0.5*k_growth*(1+np.tanh((np.subtract(time,weath_time_on_twostep)/(bb))))


weath_time_mid = 4.5 - 1.65
k_weathering_mid = 0.5*k_weath*(1+np.tanh((np.subtract(time,weath_time_mid)/bb)))
k_weathering_two_step =  k_weathering_mid

k_loT_change = k_loT*np.ones(time.size) #keep it the same
k_hiT_change = k_hiT*np.ones(time.size)

k_water_change = k_W_recycling*np.ones(time.size) #


del_steady_change = np.zeros(time.size)
del_steady_early = np.zeros(time.size)
del_steady_late = np.zeros(time.size)
del_steady_two_step = np.zeros(time.size)

k_sum = np.zeros(time.size)
k_sum_early = np.zeros(time.size)
k_sum_late = np.zeros(time.size)
k_sum_twostep = np.zeros(time.size)

for istep in range(0,time.size):
    top = np.sum([k_weathering_change[istep]*(del_graniteo[istep]-Delt_weath),\
                  k_growth_change[istep]*(del_graniteo[istep]-Delt_growth),\
                  k_hiT_change[istep]*(del_basalto-Delt_hiT_change[istep]),\
                  k_loT_change[istep]*(del_basalto-Delt_lowT),\
                  k_water_change[istep]*(del_WR-Delt_water_recycling)])
    top_two_step = np.sum([k_weathering_two_step[istep]*(del_graniteo[istep]-Delt_weath),\
                  k_growth_twostep[istep]*(del_graniteo[istep]-Delt_growth),\
                  k_hiT_change[istep]*(del_basalto-Delt_hiT_twostep[istep]),\
                  k_loT_change[istep]*(del_basalto-Delt_lowT),\
                  k_water_change[istep]*(del_WR-Delt_water_recycling)])
    top_early = np.sum([k_weathering_early[istep]*(del_graniteo[istep]-Delt_weath),\
                  k_growth_early[istep]*(del_graniteo[istep]-Delt_growth),\
                  k_hiT_change[istep]*(del_basalto-Delt_hiT),\
                  k_loT_change[istep]*(del_basalto-Delt_lowT),\
                  k_water_change[istep]*(del_WR-Delt_water_recycling)])
    top_late = np.sum([k_weathering_late[istep]*(del_graniteo[istep]-Delt_weath),\
                  k_growth_late[istep]*(del_graniteo[istep]-Delt_growth),\
                  k_hiT_change[istep]*(del_basalto-Delt_hiT_change_late[istep]),\
                  k_loT_change[istep]*(del_basalto-Delt_lowT),\
                  k_water_change[istep]*(del_WR-Delt_water_recycling)])
    
        
    k_sum[istep] = np.sum([k_weathering_change[istep],k_growth_change[istep],k_hiT_change[istep],k_loT_change[istep],k_water_change[istep]])
    k_sum_early[istep] = np.sum([k_weathering_early[istep],k_growth_early[istep],k_hiT_change[istep],k_loT_change[istep],k_water_change[istep]])
    k_sum_late[istep] = np.sum([k_weathering_late[istep],k_growth_late[istep],k_hiT_change[istep],k_loT_change[istep],k_water_change[istep]])
    k_sum_twostep[istep] = np.sum([k_weathering_two_step[istep],k_growth_twostep[istep],k_hiT_change[istep],k_loT_change[istep],k_water_change[istep]])
    
    del_steady_change[istep] = top/k_sum[istep]
    del_steady_early[istep] = top_early/k_sum_early[istep]
    del_steady_late[istep] = top_late/k_sum_late[istep]
    del_steady_two_step[istep] = top_two_step/k_sum_twostep[istep]
    
#calculate dW at for each steady state
time_new = np.linspace(0.01, 4.5,num=1000)
f1 = sp.interpolate.interp1d(time,del_steady_change)
f2 = sp.interpolate.interp1d(time,k_sum)
steady_interp = f1(time_new)
k_sum_interp = f2(time_new)
f1_late = sp.interpolate.interp1d(time,del_steady_late)
f2_late = sp.interpolate.interp1d(time,k_sum_late)
steady_interp_late = f1_late(time_new)
k_sum_interp_late = f2_late(time_new)

steady_interp_late = f1_late(time_new)
k_sum_interp_late = f2_late(time_new)

f1_early = sp.interpolate.interp1d(time,del_steady_early)
f2_early = sp.interpolate.interp1d(time,k_sum_early)
steady_interp_early = f1_early(time_new)
k_sum_interp_early = f2_early(time_new)

f1_twostep = sp.interpolate.interp1d(time,del_steady_two_step)
f2_twostep = sp.interpolate.interp1d(time,k_sum_twostep)
steady_interp_twostep = f1_twostep(time_new)
k_sum_interp_twostep = f2_twostep(time_new)


dW_middle = np.add(np.subtract(Wo,steady_interp)*np.exp(-np.multiply(time_new,k_sum_interp)),steady_interp) 
dW_early = np.add(np.subtract(Wo,steady_interp_early)*np.exp(-np.multiply(time_new,k_sum_interp_early)),steady_interp_early) 
dW_late = np.add(np.subtract(Wo,steady_interp_late)*np.exp(-np.multiply(time_new,k_sum_interp_late)),steady_interp_late) 
dW_twostep = np.add(np.subtract(Wo,steady_interp_twostep)*np.exp(-np.multiply(time_new,k_sum_interp_twostep)),steady_interp_twostep)

decay_const_low = 0.02
decay_const_high = 0.04
dW_decay_low = []#np.zeros(len(time_new))
dW_decay_high = []#np.zeros(len(time_new))
whatstep=[]



for istep in range(0,len(time_new)):
    if time_new[istep]<=1.5:
        dW_decay_low.append(np.add(np.subtract(Wo,steady_interp[-1])*np.exp(-np.multiply(time_new[istep],decay_const_low*k_sum[-1])),steady_interp[-1]))
        dW_decay_high.append(np.add(np.subtract(Wo,steady_interp[-1])*np.exp(-np.multiply(time_new[istep],decay_const_high*k_sum[-1])),steady_interp[-1]))
        temp_Wo_low = dW_decay_low[istep]
        temp_Wo_high = dW_decay_high[istep]
        whatstep.append(istep)
knickpoint = whatstep[-1]
time_late = np.flip(time_new[-1] - time_new[knickpoint+1:],0) 
low_test = []
high_test = []
for istep in range(0,len(time_late)):
    dW_decay_low.append(np.add(np.subtract(temp_Wo_low,steady_interp[-1])*np.exp(-np.multiply(time_late[istep],0.4*k_sum[-1])),steady_interp[-1]))
    dW_decay_high.append(np.add(np.subtract(temp_Wo_high,steady_interp[-1])*np.exp(-np.multiply(time_late[istep],0.5*k_sum[-1])),steady_interp[-1]))
    low_test.append(np.add(np.subtract(temp_Wo_low,steady_interp[-1])*np.exp(-np.multiply(time_late[istep],0.03*k_sum[-1])),steady_interp[-1]))

#%% ----create regular dataset ------------- %%# 
#Zircon age read

age_data = pd.read_excel('zircon_ages.xlsx')

#%%Plot

# data_raw = pd.read_csv('Elevation and depth.csv')

plt.figure(); plt.clf()
binwidth = 50 # 50 million year time bins

plt.hist(age_data['Best Age (Ma)'],
         bins=np.arange(min(age_data['Best Age (Ma)']), max(age_data['Best Age (Ma)']) + binwidth, binwidth),
         color='k')
plt.xlim([0,4500])
plt.gca().invert_xaxis()
plt.xlabel('Age (Ma)')
plt.ylabel('Num. of zircons')
#%% 
#plots 
time_labels = ['4.5','4','3.5','3','2.5','2','1.5','1','0.5','0']
time_ticks = np.linspace(0,4.5,10)
#plt.close('all')
f, (ax1) = plt.subplots(1, 1, sharey=True)
ax1 = plt.subplot(1,1,1)
ax1.plot(time,k_growth_change,'k:')
ax1.plot(time,k_weathering_change,'k--')
ax1.plot(time,k_loT_change,'k-.')
ax1.plot(time,k_hiT_change,'k-')
ax1.plot(time,k_water_change,'k.-')
ax1.legend(['Continental recycling','Continental weathering','low T alteration','high T alteration','Water recycling']\
           ,bbox_to_anchor=(0.01, 0.93), loc=2, borderaxespad=0.,fontsize=8)
plt.xlabel('Age (Ga)'); plt.ylabel('Ocean $\delta^{18}$O')
locs, labels = plt.xticks()           # Get locations and labels

plt.xticks(time_ticks, time_labels)  # Set locations and labels
plt.xticks(time_ticks, time_labels)  
ax1.set_xlim([0,4.5])
plt.ylabel('Rate (Gyr $^{-1}$)')


#%%-----Seawater and zircon ages together-------#
fig2, (ax2) = plt.subplots(1, 1, sharey=True)

ax2 = plt.subplot(1,1,1) # subplot(2,1,2) is now active

# plt.figure();plt.clf()
ax2.hist(np.subtract(4.5,age_data['Best Age (Ma)']/1000),
         bins=np.arange(min(age_data['Best Age (Ma)'])/1000, max(age_data['Best Age (Ma)'])/1000 + binwidth/1000, binwidth/1000),
         color='k',alpha=0.25)
plt.xlim([0,4.500])
# plt.gca().invert_xaxis()
plt.xlabel('Age (Ma)')
ax2.set_ylabel('Num. of zircons')

# fig2.set_size_inches(12, 8.5)
#ax2.set_xlim([0,4.5])
plt.xticks(time_ticks, time_labels)  # Set locations and labels

#plt.yticks([])
# plt.title('Ocean $\delta^{18}$O constraints')
 # ax2.plot(time_new,dW_decay,'k--',linewidth=2)
# ax2.plot(time_new,dW_early,'k-.',linewidth=2)
ax3 = ax2.twinx()
ax3.plot(time_new,dW_middle,'k:',linewidth=2)
ax3.plot(time_new,dW_late,'k-',linewidth=2)
ax3.plot(time_new,dW_twostep,'k--',linewidth=2)
plt.xlabel('Age (Ga)'); ax3.set_ylabel('Seawater $\delta^{18}$O')

new_data = [0.2, 0.72, 0.97,3.46] 
new_error = [0.3, 0.2, 0.03, 0.16]
new_ages = np.subtract(4.5,[1.72, 1.89, 2.682,2.735]) #Hydrothermal cell inversion results

plt.errorbar(new_ages,new_data,yerr=new_error,fmt='ko',markersize=12,elinewidth=4,capsize=5)
#'Hadean  emergence',
ax3.legend(['Late Archean emergence','Neoproterozoic emergence','Two stage emergence'])#,'Early emergence','Late emergence'])
other_color = np.divide([219,168,133],255)
our_color = np.divide([144,110,110],255)
Pope = patches.Rectangle((4.5-3.75,0.8),0.1,3,linewidth=1,edgecolor='k',facecolor=other_color)
ax3.add_patch(Pope)
Ours = patches.Rectangle((4.5-3.19,2.7),0.1,0.7,linewidth=1,edgecolor='k',facecolor=other_color)
ax3.add_patch(Ours)
#−1.33 ± 0.98‰
Hodel = patches.Rectangle((4.5-0.71,-2.28),0.1,1.95,linewidth=1,edgecolor='k',facecolor=other_color)
ax3.add_patch(Hodel)
#0+-2
Ordo = patches.Rectangle((4.5-0.5,-2),0.1,4,linewidth=1,edgecolor='k',facecolor=other_color) 
ax3.add_patch(Ordo)
Samail =patches.Rectangle((4.5-0.085,-1.4),0.1,2,linewidth=1,edgecolor='k',facecolor=other_color) 
ax3.add_patch(Samail)

mag_oc = patches.Rectangle((0,6),0.1,2,linewidth=1,edgecolor='k',facecolor=other_color) 
ax3.add_patch(mag_oc)

# a_em =patches.Rectangle((4.5-2.5,-3),.03,10,linewidth=1,edgecolor='k',facecolor='xkcd:cloudy blue',alpha=0.4) 
# ax2.add_patch(a_em)
# b_em =patches.Rectangle((4.5-0.7,-3),.13,10,linewidth=1,edgecolor='k',facecolor='xkcd:cloudy blue',alpha=0.4) 
# ax2.add_patch(b_em)
# c_em =patches.Rectangle((4.5-3.5,-3),.03,10,linewidth=1,edgecolor='k',facecolor='xkcd:cloudy blue',alpha=0.4) 
# ax2.add_patch(c_em)
# d_em =patches.Rectangle((4.5-3.2,-3),.03,10,linewidth=1,edgecolor='k',facecolor='xkcd:cloudy blue',alpha=0.4) 
# ax2.add_patch(d_em)
# e_em =patches.Rectangle((4.5-3,-3),.03,10,linewidth=1,edgecolor='k',facecolor='xkcd:cloudy blue',alpha=0.4) 
# ax2.add_patch(e_em)
plt.ylim([-2.5,7])
plt.xlim([0,4.5])


