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
#%% Seawater oxygen isotope exchange model - do not change core parameters!
t = 4.4 #time in Gyr

Wo = 7 #original seawater d18O from magma ocean cooling constraints, not reasonable to discard
W_SS = -1 #steady state, no cryosphere
num_steps = 100 #num of initial model steps
time = np.linspace(0,4.5,num=num_steps) #sample every 250 myr
weath_time_on_twostep = 4.5-2.8 #in Ga
weath_time_on = 4.5-2.7 #Hadean-end Archean
weath_time_early = 4.5-4.43 #Hadean-Early Archean
weath_time_late = 4.5-0.9 #Hadean-Proterozoic
Time = 4.5-time

#%%
# These can be modified and tuned to reproduce chemical sediment behavior with secular increase in oceanic d18O
# rate constants in Gyr-1, from Muehlenbachs, 1998
k_weath = 8 # 3200*(4.5-time) #220-19*((time-1)*(time-1)) #nominal 8 continental weathering - started with 8 - strongly affects model at high to modest values, dependent strongly on timing of continental emergence, creates rapid dropoff in d18O seawater for Hadean-Archean weathering scenarios
k_growth = 1.2 #nominal 1.2 continental growth - started with 1.2 - weakly affects model
k_hiT = 14.6*(time*time/4) + 3 # 110*time*time #14.6 #np.exp(4*time)#1100*(4.5 - time)*(4.5-time)*(4.5-time) +14.6 # 14.6 # ! vary decreasing with time linear ! 110*((4.5-time)) #nominal 14.6 high temperature seafloor - started with 14.6 - weakly affects model
k_loT = 1.7*Time*Time + 2 # 110*Time*Time*Time #1.7 #np.exp(4*Time)#100000*(time*time*time) # ! vary increasing with time linear ! nominal 1.7 low temp seafloor/seafloor weathering - started with 1.7 - weakly affects model
k_W_recycling = 0.6 # 1+(1000*time) #nominal 0.6 water recycling at subduction zones - started with 0.6 - strongly affects model at high values, creates upwards inflection towards terminal d18O seawater value

#fractionations (permil) btwn rock and water, from Muehlenbachs (1998) except weathering, which we tuned to reproduce -1 permil ocean

Delt_weath =  13  #mueh = 9.6 nominal 13, newer 17 - started with 13, then 
Delt_growth =  9.8 #mueh = 9.8 - started with 9.8, then 
Delt_hiT_mid =  1.5 #meuh = 4.1 - started with 1.5, then 
Delt_hiT = 7 #4.1 #unknown provenance, maybe best fit? - started with 4.1, then 
#Delt_hiT_mid = np.zeros(len(time)) # unknown provenance - not used
Delt_lowT = 14 #mueh = 9.3 - started with 9.3, then 
Delt_water_recycling = 2.5 # mueh = 2.5 - started with 2.5, then 
    

#calculate steady state in 250 myr increments - unchanged for now
del_graniteo = 4.5 + 0.03666*np.exp(time) #np.linspace(5.5,7.8,num=num_steps) #modern 7.8 constant, can rise from 5.5 mafic early continent composition

del_basalto = 5.5  # 5.5 leave unchanged
del_WR = 7 # 7 leave unchanged for now
bb= 0.1 # 0.1 leave unchanged

#%%

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
    
#%%
#read and plot C and O isotopes from Prokoph and older papers for presentation
data_raw = pd.read_csv('Prokoph_compilation.csv')

#extracts 18O data from .csv file for calcite and dolomite and compares with age
del18Occ = pd.Series(data_raw['d18Ocalcite  PDB']).values
del18Odo = pd.Series(data_raw['d18Odolomite PDB']).values
age = pd.Series(data_raw['Age (Ma)']).values
#
#remove nans
nan_cc_idx = np.argwhere(np.isnan(del18Occ))
nan_do_idx = np.argwhere(np.isnan(del18Odo))

del18Occ= np.delete(del18Occ,nan_cc_idx,axis=0)
age_cc = np.delete(age,nan_cc_idx,axis=0)

#convert to SMOW 


del18Odo= np.delete(del18Odo,nan_do_idx,axis=0)
age_do = np.delete(age,nan_do_idx,axis=0)


agesteps = np.linspace(500,3400,10)
cctrend = np.interp(agesteps,age_cc,del18Occ) #calcite trend
dotrend = np.interp(agesteps,age_do,del18Odo) #dolomite trend 
plt.figure(num=1); plt.clf()
plt.plot(age_cc,del18Occ,'ko')
plt.plot(age_do,del18Odo,'ks')
plt.plot(agesteps,cctrend,'r--')
plt.plot(agesteps,dotrend,'b--')
plt.xlabel('Age (Ma)')
plt.ylabel('$\delta^{18}$O - carbonate')

#%%
#Veizer et al. Precambrian carbonate data
data_rawV = pd.read_csv('veizer_pC.csv') #
del18OccV = pd.Series(data_rawV['d18Occ']).values
del18OdoV = pd.Series(data_rawV['d18Odo']).values
del18OdoV = pd.to_numeric(del18OdoV, errors='coerce') 
ageV = pd.Series(data_rawV['age']).values

#Veizer et al. 1997 Oxygen Isotope Evolution of Phanerozoic Seawater? Phanerozoic carbonate data
data_rawVp = pd.read_csv('vezier_phan.csv') #
ageVp = pd.Series(data_rawVp['Age']).values
ageVp = pd.to_numeric(ageVp, errors='coerce')
del18OccVp = pd.Series(data_rawVp['d18O']).values

del18OccVpsmow = del18OccVp*1.03092+30.92

#remove nans
nan_cc_idxV = np.argwhere(np.isnan(del18OccV))
nan_do_idxV = np.argwhere(np.isnan(del18OdoV))

del18OccV= np.delete(del18OccV,nan_cc_idxV,axis=0)
age_ccV = np.delete(ageV,nan_cc_idxV,axis=0)


del18OdoV= np.delete(del18OdoV,nan_do_idxV,axis=0)
age_doV = np.delete(ageV,nan_do_idxV,axis=0)

#convert to SMOW 
del18OccVsmow = del18OccV*1.3086+30.86
del18OdoVsmow = del18OdoV*1.3086+30.86

all_data = np.concatenate((del18OccVsmow,del18OdoVsmow,del18OccVpsmow))
all_ages = np.concatenate((age_ccV,age_doV,ageVp))

agesteps = np.linspace(500,3400,1000) #starting (youngest) age, ending (earliest) age, timesteps for x axis
cctrendV = np.interp(agesteps,age_ccV,del18OccVsmow)
dotrendV = np.interp(agesteps,age_doV,del18OdoVsmow)
all_trend = np.interp(agesteps,all_ages,all_data)

#%% ----create regular dataset ------------- %%# 
#Zircon age read

age_data = pd.read_excel('zircon_ages.xlsx')

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
plt.xlabel('Age (Ga)') #said Ma originally, messed up x axis 
ax2.set_ylabel('Num. of zircons')

# fig2.set_size_inches(12, 8.5)
#ax2.set_xlim([0,4.5])
plt.xticks(time_ticks, time_labels)  # Set locations and labels

#plt.yticks([])
# plt.title('Ocean $\delta^{18}$O constraints')
# ax2.plot(time_new,dW_decay,'k--',linewidth=2)
ax3 = ax2.twinx()
ax3.plot(time_new,dW_early,'k-.',linewidth=2)
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
plt.ylim([-10,10])
plt.xlim([0,4.5])

#%%
#plot seawater delta18O hydrothermal model outputs against carbonate seawater delta18O reconstruction
#fig2=plt.figure(num=3); plt.clf()
fig2, (ax2) = plt.subplots(1, 1, sharey=True)
time_labels = ['4.5','4','3.5','3','2.5','2','1.5','1','0.5','0']
#time_ticks = np.linspace(0,4.5,10)
plt.xticks(time_ticks, time_labels)  # Set locations and labels
#xticks = [4.5,4,3.5,3,2.5,2,1.5,1,0.5,0,]
ax2.plot(age_ccV,del18OccVsmow,'k.',fillstyle='none')
ax2.plot(ageVp,del18OccVpsmow,'k.',fillstyle='none')
ax2.plot(age_doV,del18OdoVsmow,'k.',fillstyle='none')
ax=plt.gca()
ax.invert_xaxis()
x1=0;x2=4500
y1=-1;y2=-1 #corrected from 0,0 to generate constant line of ice-free ocean
#plot rough fit lines to carbonate data and seawater d18O extrapolation (assuming constant fractionation in 30-35 C range)
ax2.plot([x1, x2], [y1, y2], color='b', linestyle='-', linewidth=3)
ax2.plot([x1, x2], [30, 8], color='r', linestyle='--', linewidth=3)
ax2.plot([x1, x2], [-1, -30], color='b', linestyle='-.', linewidth=3) #corrected from [0,-30] to account for ice-free ocean 
#plot hydrothermal d18O seawater model lines for twostep, late Archean, and Neoprot emergence
ax2.plot(4500-time_new*1000,dW_middle,'k:',linewidth=2) #these three plotted in reverse for reasons unknown, so must be subtracted from 4500 (total time) to achieve proper order
ax2.plot(4500-time_new*1000,dW_late,'k-',linewidth=2) #"
ax2.plot(4500-time_new*1000,dW_early,'k-.',linewidth=2)
ax2.plot(4500-time_new*1000,dW_twostep,'k--',linewidth=2) #"
#display variables used in each plot-generation simulation, k = rate constants per Gy and Delt = fractionations (permil) between rock and water
plt.text(2500,-3.5,'Modern Ocean',color='b')
plt.text(4400,30,'k_weath: '+str(k_weath))
plt.text(4400,28,'k_growth: '+str(k_growth))
plt.text(4400,26,'k_hiT: '+str(k_hiT))
plt.text(4400,24,'k_loT: '+str(k_loT))
plt.text(4400,22,'k_W_recycling: '+str(k_W_recycling))
plt.text(4400,20,'Delt_weath: '+str(Delt_weath))
plt.text(4400,18,'Delt_growth: '+str(Delt_growth))
plt.text(4400,16,'Delt_hiT_mid: '+str(Delt_hiT_mid))
plt.text(4400,14,'Delt_hiT: '+str(Delt_hiT))
plt.text(4400,12,'Delt_lowT: '+str(Delt_lowT))
plt.text(4400,10,'Delt_WR: '+str(Delt_water_recycling))
#complete plotting
plt.xlim([x1,x2])
plt.xticks([4500,4000,3500,3000,2500,2000,1500,1000,500,0])
ax.set_xticklabels(time_labels)
plt.xlabel('Age (Ga)'); plt.ylabel('Seawater $\delta^{18}$O')
ax=plt.gca()
ax.invert_xaxis()
fig2.set_size_inches(11.5, 8.5)
#plt.ylim([-30,30]) #originally [-2.5, 7]
#plt.xlim([0,4.5])
#%%
fig2, (ax2) = plt.subplots(1, 1, sharey=True)

#ax2 = plt.subplot(1,1,1) # subplot(2,1,2) is now active
time_labels = ['4.5','4','3.5','3','2.5','2','1.5','1','0.5','0']
time_ticks = np.linspace(0,4.5,10)
plt.xticks(time_ticks, time_labels)  # Set locations and labels

#plt.yticks([])
# plt.title('Ocean $\delta^{18}$O constraints')
# ax2.plot(time_new,dW_decay,'k--',linewidth=2)
ax2.plot(time_new,dW_early,'k-.',linewidth=2)
ax2.plot(time_new,dW_middle,'k:',linewidth=2)
ax2.plot(time_new,dW_late,'k-',linewidth=2)
ax2.plot(time_new,dW_twostep,'k--',linewidth=2)
plt.xlabel('Age (Ga)'); plt.ylabel('Seawater $\delta^{18}$O')
#fig5=plt.figure(num=2); plt.clf() 
#plots 
#time_labels = ['4.5','4','3.5','3','2.5','2','1.5','1','0.5','0']
#time_ticks = np.linspace(0,4500,500)

#plot hydrothermal delt18O model outputs on same figure
#ax5 = ax.twinx()
#ax5.plot(time_new,dW_middle,'k:',linewidth=2)
#ax5.plot(time_new,dW_late,'k-',linewidth=2)
#ax5.plot(time_new,dW_twostep,'k--',linewidth=2)
#plt.xlabel('Age (Ga)'); ax5.set_ylabel('Seawater $\delta^{18}$O')
plt.ylim([-30,30]) #originally [-2.5, 7]
plt.xlim([0,4.5])

#%%
#prints sensitivity parameters for delta 18O seawater values for time of interest (using late emergence as a typical history, can vary)
time_int_max = 250 # 4500 - age of interest (Ma) for early (Hadean) stable ocean value - typically 250
max_norm_dW = dW_late[time_int_max]

time_int_min = 998 # 4500 - age of interest (Ma) for recent ocean value - typically 998
min_norm_dW = dW_late[time_int_min]

#finds range of interest between earliest stable seawater value and modern value, range of variation
range_int = max_norm_dW - min_norm_dW

range_int_std = 4.249427524849613 # value of range_int with standard model parameters
Sgma = range_int/range_int_std

median_int = (max_norm_dW + min_norm_dW)/2
median_int_std = 1.0220318181347245 # median_int value with standard model parameters
Diff_med_int = median_int - median_int_std

#calculates net sensitivity parameter for response of model output range and mean to 10-factor changes in given parameter(s)
net_sensitivity = 10*(abs(np.log(Sgma))) + abs(Diff_med_int)

print("max_dW: ", max_norm_dW)
print("min_dW: ", min_norm_dW)

print("range: ", range_int)
print("median_int: ", median_int)

print("Sigma sensitivity: ", Sgma)
print("DeltaMedian sensitivity value: ", Diff_med_int)
print("Net Sensitivity: ", net_sensitivity)


# max_dW_std at time_new = 250: 3.1467455805595312
# min_dW_std at time_new = 998: -1.102681944290082






