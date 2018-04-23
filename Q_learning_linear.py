import numpy
import sys
import numpy
import itertools
from itertools import chain, combinations
import math
import copy
from random import randint
import random

N_m=10
N_s=int(math.pow(2,N_m))
N_a=N_m+1
#all_actions=[0,1,2,3,4]
gamma=0.9
alpha=0.01
def powerset(iterable):
  xs = list(iterable)
  return chain.from_iterable( combinations(xs,n) for n in range(len(xs)+1) )

N_m_array=[]
for i in range(0,N_m):
    N_m_array.append(i)
#all_basis=list(powerset(set(N_m_array)))
#print all_basis
def Rewards(x,a):
    reward=0
    #print x
    for i in range(0,N_m-1):
        reward=reward+x[i]

    reward=reward+2*x[N_m-1]
    return reward


def States(index):
    all_states=list(itertools.product([0, 1], repeat=N_m))
    x=all_states[index]
    return x


def CPD(i, x_hat_i, y_p, a):
    if x_hat_i == 0:
        p = (1 - probability_table(i, y_p, a))
    else:
        p = probability_table(i, y_p, a)
    return p


# this probability table is only for the two parent model
def probability_table(i, y_p, a):
    if a == i:
        prob = 1
    else:
        if y_p[0] == 0 and y_p[1] == 0:
            prob = 0.05
        if y_p[0] == 0 and y_p[1] == 1:
            prob = 0.5
        if y_p[0] == 1 and y_p[1] == 0:
            prob = 0.09
        if y_p[0] == 1 and y_p[1] == 1:
            prob = 0.9
    return prob

def parent_indices(i):
    p=numpy.zeros(2)
    if i==0:
       p[0]=0
       p[1]=N_m-1
    else:
       p[0]=i-1
       p[1]=i
    return p.astype(int)

#initialize feature weights to 0
theta=numpy.zeros(N_m*N_a +1)

def compute_Q(theta,state,action):
    Q=0
    feature_vector=numpy.zeros(len(theta))
    feature_vector[0]=1
    for j in range(0,N_a):
        for i in range(0,N_m):
            if action ==j:
               feature_vector[j*N_m+i+1]= math.pow(-1,state[i-1])
            else:
               feature_vector[j*N_m+i+1]=0
            Q = Q + theta[j*N_m+i]*feature_vector[j*N_m+i]
    #print feature_vector
    #print theta
    return Q , feature_vector
#print theta


epsilon=0.05

num_epochs=100

for i in range(0,1000):
    random_state = numpy.random.randint(2, size=N_m)
    random_state = numpy.array([random_state])
    if i==0:
       eval_set=copy.deepcopy(random_state)
    else:
       eval_set= numpy.concatenate((eval_set,random_state))




eval_set=numpy.matrix(eval_set)
#eval_set=numpy.matrix('0,0,0,0,0,0;0,1,0,0,1,1;1,0,1,0,1,1;1,1,1,1,1,0;0,0,0,1,0,1')
curr_state = numpy.zeros(N_m)
diff=numpy.zeros(num_epochs)
eval_Q_sum=numpy.zeros(num_epochs)

estim_Q= numpy.zeros(num_epochs)
pred_Q= numpy.zeros(num_epochs)
for epochs in range(0,num_epochs):
    #choose an action with epsilon greedy strategy
    maxQ=0
    a=0
    for actions in range(0,N_a):
        computed_Q,_= compute_Q(theta,curr_state,actions)
        if computed_Q> maxQ :
            maxQ=computed_Q
            a=actions
    if (random.random() < epsilon):  # choose random action
        action_choice = numpy.random.randint(0, N_a)
    else:  # choose best action from Q(s,a) values
        action_choice = a

    #print action_choice
    # observe the next state and the reward
    next_state = numpy.zeros(N_m)
    for j in range(0,N_m):
        parent_nodes=parent_indices(j)
        parent_values=[]
        for k in range(0,len(parent_nodes)):
            parent_values.append(curr_state[parent_nodes[k]])

        prob_next_state= CPD(j, 1, parent_values, action_choice)
        #choose next state with that probability
        next_state[j]=numpy.random.choice([1,0], 1, p=[prob_next_state,1-prob_next_state])


    R=Rewards(next_state,action_choice)
    #print R
    #compute max Q over a starting from this next state
    max_Q=-100000
    for actions in range(0,N_a):
        Q_value,dummy=  compute_Q(theta,next_state,actions)
        if  Q_value  > max_Q:
            max_Q = Q_value

    #compute estimated Q: R+gamma*max Q
    estimated_Q=R+gamma*max_Q
    # compute difference and store in a vector, plot at the end
    predicted_Q,feature_vector= compute_Q(theta,curr_state,action_choice)
    diff[epochs] =   estimated_Q - predicted_Q
    estim_Q[epochs]=estimated_Q
    pred_Q[epochs] =predicted_Q
    #print   estimated_Q - predicted_Q
    #update patameters in gradient decsent
    #decrease alpha incrementally
    alpha=1000/float(4000 +epochs)
    #print alpha
    for i in range(0, len(theta)):
        theta[i]=theta[i]+alpha*diff[epochs]*feature_vector[i]

    curr_state=copy.deepcopy(next_state)
    print theta





#print theta





eval_sum = 0

max_Q=0

for eval_s in range(0, 1000):
    for actions in range(0,N_a):
        computed_Q,_= compute_Q(theta,eval_set[eval_s].reshape(N_m,1),actions)
        if computed_Q> max_Q :
                max_Q= computed_Q

    eval_sum += max_Q

print eval_sum/1000
eval_Q_sum[epochs]=eval_sum/1000


import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.plot(numpy.abs(eval_Q_sum),'-g')
#plt.plot(estim_Q,'-gs',ms=0,linewidth=2)
#plt.plot(pred_Q,'-bs',ms=0, linewidth=2)
plt.ylabel('Average action value (Q)',fontsize=16)
plt.xlabel('Training iterations',fontsize=16)
plt.show()
