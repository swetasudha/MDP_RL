import numpy
import sys
import numpy
import itertools
from itertools import chain, combinations
import math
import copy
from random import randint
import random
import tensorflow as tf
from keras import backend as K

epochs = 10000
gamma = 0.9
epsilon = 0.1
batchSize = 200
buffer = 300
N_m=10
N_a=11
LEARNING_RATE=0.01
HUBER_LOSS_DELTA=1


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

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *



def huber_loss(y_true, y_pred):
    err = y_true - y_pred

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)

    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(

    return K.mean(loss)

model = Sequential()
hidden_layer_size=int(N_m+N_a)/2

model.add(Dense(units=hidden_layer_size, activation='relu', input_shape=(N_m,)))
model.add(Dense(units=N_a, activation='linear'))

opt = RMSprop(lr=LEARNING_RATE)
model.compile(loss=huber_loss, optimizer=opt)


print model.layers[0].get_weights()
print model.layers[1].get_weights()


# here is the target model
target_model = Sequential()


target_model.add(Dense(units=hidden_layer_size, activation='relu', input_shape=(N_m,)))
target_model.add(Dense(units=N_a, activation='linear'))

opt = RMSprop(lr=LEARNING_RATE)
target_model.compile(loss=huber_loss, optimizer=opt)

target_model.set_weights(model.get_weights())


for i in range(0,100):
    random_state = numpy.random.randint(2, size=N_m)
    random_state = numpy.array([random_state])
    if i==0:
       eval_set=copy.deepcopy(random_state)
    else:
       eval_set= numpy.concatenate((eval_set,random_state))




eval_set=numpy.matrix(eval_set)

#eval_set=numpy.matrix('0,0,0,0,0,0;0,1,0,0,1,1;1,0,1,0,1,1;1,1,1,1,1,0;0,0,0,1,0,1')


eval_Q_sum=numpy.zeros(epochs)

replay = []
# stores tuples of (S, A, R, S')
h = 0


#initializing the state
state = numpy.zeros(N_m)
for i in range(epochs):
    if i%50==0:
        target_model.set_weights(model.get_weights())

    # We are in state S
    # Let's run our Q function on S to get Q values for all possible actions
    qval = model.predict(state.reshape(1, N_m), batch_size=1)
    a=numpy.argmax(qval)
    # epsilon greedy part
    if (random.random() < epsilon):  # choose random action
        action = numpy.random.randint(0, N_a)
    else:  # choose best action from Q(s,a) values
        action = a
    # Take action, observe new state S'

    new_state = numpy.zeros(N_m)
    action_choice=action
    for j in range(0, N_m):
        parent_nodes = parent_indices(j)
        parent_values = []
        for k in range(0, len(parent_nodes)):
            parent_values.append(state[parent_nodes[k]])

        prob_next_state = CPD(j, 1, parent_values, action_choice)
        # choose next state with that probability
        new_state[j] = numpy.random.choice([1, 0], 1, p=[prob_next_state, 1 - prob_next_state])

    reward = Rewards(new_state, action_choice)

    # Experience replay storage
    if (len(replay) < buffer):  # if buffer not filled, add to it
        replay.append((state, action, reward, new_state))
    else:  # if buffer full, overwrite old values
        if (h < (buffer - 1)):
            h += 1
        else:
            h = 0
        replay[h] = (state, action, reward, new_state)
        # randomly sample our experience replay memory
        minibatch = random.sample(replay, batchSize)
        X_train = []
        y_train = []
        for memory in minibatch:
            # Get max_Q(S',a)
            old_state, action, reward, new_state = memory
            old_qval = model.predict(old_state.reshape(1, N_m), batch_size=1)
            newQ = target_model.predict(new_state.reshape(1, N_m), batch_size=1)
            maxQ = numpy.max(newQ)
            y = numpy.zeros((1, N_a))
            y[:] = old_qval[:]
            update = (reward + (gamma * maxQ))
            y[0][action] = update
            X_train.append(old_state.reshape(N_m, ))
            y_train.append(y.reshape(N_a, ))

        X_train = numpy.array(X_train)
        y_train = numpy.array(y_train)

        model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=0)


        eval_sum=0

        for eval_s in range(0,100):
            Q_pred=target_model.predict(eval_set[eval_s].reshape(1, N_m), batch_size=1)
            eval_sum+=numpy.max(Q_pred)
        if i % 100 == 0:
           print eval_sum/100
        eval_Q_sum[i] = eval_sum / 100
        state = new_state

        #if epsilon > 0.1:  # decrement epsilon over time
        epsilon -= 0.001
        #print epsilon



for eval_s in range(0,100):
    Q_pred = model.predict(eval_set[eval_s].reshape(1, N_m), batch_size=1)
    print Q_pred
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.plot(numpy.abs(eval_Q_sum),'-g')
#plt.plot(estim_Q,'-gs',ms=0,linewidth=2)
#plt.plot(pred_Q,'-bs',ms=0, linewidth=2)
plt.ylabel('Average action value (Q)',fontsize=16)
plt.xlabel('Training iterations',fontsize=16)
plt.show()





