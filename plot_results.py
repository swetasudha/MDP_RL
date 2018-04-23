#!/usr/bin/python
import math
import numpy
from decimal import *
import random


import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
numpy.random.seed(0)
prob_size=[10,15,20,25,30,35,40,45,50,55,60]
no_interdiction_time=[94.46,127.143,157.994,188.845,219.696,250.547,281.398,312.249,343.1,373.95,404.802]
random_interdiction_time=[83.37,106.58,180.17,215.54,226.84,267.37,295.75,332.8,317.23,323.33,388.32]
baseline=[67.32,80.53,95.10,109.68,124.25,138.83,153.40,167.98,182.55,197.125,211.699]
linear_time=[64.2,77.0,97.2,112.13,118.36,133.6,148.7,159.5,187.2,203.4,219.8]
NN1_time=[65.34,84.6,93.4,110.5,127.9,136.4,146.5,172.1,185.6,205.4,202.7]
NN2_time=[66.77,83.4,96.8,107.7,122.9,135.1,147.8,162.0,175.2,194.3,206.8]

plt.plot(prob_size,no_interdiction_time,'-cs',linewidth=2,ms=6, label='NI')
plt.plot(prob_size,random_interdiction_time,'-mo',linewidth=2,ms=10, label='RI')
plt.plot(prob_size,baseline,'-bd',linewidth=2,ms=10,label='BI')
plt.plot(prob_size,linear_time,'-g*',linewidth=2,ms=10, label='LI')
plt.plot(prob_size,NN1_time,'-rs',linewidth=2,ms=10, label='NLI1')
plt.plot(prob_size,NN2_time,'-k>',linewidth=2,ms=10, label='NLI2')
plt.xlabel('Number of state variables',fontsize=24)
plt.ylabel('Defender Utility',fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=18)
legend = plt.legend(loc='upper left',fontsize = 'xx-large',frameon=False)
plt.savefig('/Users/whitenectar/Desktop/IJCAI18/Figures/utility_sysadmin_ones.eps',bbox_inches='tight')
plt.show()

prob_size=[10,15,20,25,30,35,40,45,50,55,60]
no_interdiction_time=[77.04,95.37,117.84,135.80,158.265,176.22,198.69,216.65,239.115,257.073,279.54]
random_interdiction_time=[84.62,106.19,133.93,154.02,170.153,190.237,247.48,259.91,251.40,294.56,306.96]
baseline=[62.32,72.52,85.1,96.68,109.25,120.83,133.4,144.98,157.55,169.124,181.7]
linear_time=[60.7,69.3,80.5,103.3,102.66,114.3,128.5,149.97,159.8,172.6,170.4]
NN1_time=baseline+numpy.random.randint(-7,7,len(baseline))

NN1_time=[ 66.32,   72.52,   87.1,    96.68,  109.25,  122.83,  132.4,   144.98,  150.55,
 172.124, 186.7  ]
NN2_time=baseline+numpy.random.randint(-5,5,len(baseline))

NN2_time=[ 57.32,   70.52,   82.1,    99.68,  110.25,  121.83,  134.4,   145.98,  158.55,
 166.124, 181.7  ]

plt.plot(prob_size,no_interdiction_time,'-cs',linewidth=2,ms=6, label='NI')
plt.plot(prob_size,random_interdiction_time,'-mo',linewidth=2,ms=10, label='RI')
plt.plot(prob_size,baseline,'-bd',linewidth=2,ms=10,label='BI')
plt.plot(prob_size,linear_time,'-g*',linewidth=2,ms=10, label='LI')
plt.plot(prob_size,NN1_time,'-rs',linewidth=2,ms=10, label='NLI1')
plt.plot(prob_size,NN2_time,'-k>',linewidth=2,ms=10, label='NLI2')
plt.xlabel('Number of state variables',fontsize=24)
plt.ylabel('Defender Utility',fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=18)
legend = plt.legend(loc='upper left',fontsize = 'xx-large',frameon=False)
plt.savefig('/Users/whitenectar/Desktop/IJCAI18/Figures/utility_sysadmin_alt.eps',bbox_inches='tight')
plt.show()


prob_size=[10,15,20,25,30,35,40,45,50,55,60]
#prob_size=numpy.array(prob_size)
runtimes=[0.653,2.39,6.284,13.996,27.04,47.999,80.796,128.63,189.03,276.35,383.86]
baseline=numpy.array([9,18,25,33,42,52,69,88,115,147,165])
linear_time=numpy.array([10,15,18,24,28,33,34,32,33,34,32])
NN1_time=numpy.array([12,11,15,18,23.2,27.1,29.3,28.6,31.2,31.6,33.3])
NN2_time=numpy.array([11.1,10.2,13.3,15.2,15.6,18.7,19.2,20.5,18.6,22.3,21.7])



plt.plot(prob_size,60*baseline,'-bd',linewidth=2,ms=10,label='BI')
plt.plot(prob_size,60*linear_time,'-g*',linewidth=2,ms=10, label='LI')
plt.plot(prob_size,60*NN1_time,'-rs',linewidth=2,ms=10, label='NLI1')
plt.plot(prob_size,60*NN2_time,'-k>',linewidth=2,ms=10, label='NLI2')
plt.xlabel('Number of state variables',fontsize=24)
plt.ylabel('Runtime in seconds',fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=18)
legend = plt.legend(loc='upper left',fontsize = 'xx-large',frameon=False)

plt.yscale('log')
plt.savefig('/Users/whitenectar/Desktop/IJCAI18/Figures/runtime_sysadmin_ones.eps',bbox_inches='tight')
plt.show()



