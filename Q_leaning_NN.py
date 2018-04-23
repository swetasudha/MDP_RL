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

epochs = 1000
gamma = 0.9
epsilon = 0.05
batchSize = 200
buffer = 500
N_m=50
N_a=51


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



#the NN model
tf.reset_default_graph()
#These lines establish the feed-forward part of the network used to choose actions
inputs = tf.placeholder(tf.float32, [None,N_m])
hidden_layer_size=int(N_m+N_a)/2
W1 = tf.Variable(tf.random_uniform([N_m, hidden_layer_size],0,0.01))
b1 = tf.Variable(tf.random_uniform([hidden_layer_size],0,0.01))
W2 = tf.Variable(tf.random_uniform([hidden_layer_size,N_a],0,0.01))
b2 = tf.Variable(tf.random_uniform([N_a],0,0.01))
hidden_out = tf.add(tf.matmul(inputs, W1), b1)
hidden_out = tf.nn.relu(hidden_out)
Q_out =tf.add(tf.matmul(hidden_out, W2), b2)
#Q_out = tf.nn.relu(hidden_out)
predict = tf.argmax(Q_out,1)
# predict is for the best action, Q_out is the vector of output Q values

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[None,N_a],dtype=tf.float32)
#Next Q is the actual score data for training

loss = tf.reduce_sum(tf.square(nextQ - Q_out))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
updateModel = trainer.minimize(loss)

for i in range(0,1000):
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
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #initializing the state
    state = numpy.zeros(N_m)
    for i in range(epochs):
        # We are in state S
        # Let's run our Q function on S to get Q values for all possible actions
        a, _=sess.run([predict, Q_out], feed_dict={inputs: state.reshape(1, N_m)})

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
                _,old_qval = sess.run([predict, Q_out], feed_dict={inputs: old_state.reshape(1, N_m)})
                _,newQ =sess.run([predict, Q_out], feed_dict={inputs: new_state.reshape(1, N_m)})
                maxQ = numpy.max(newQ)
                y = numpy.zeros((1, N_a))
                y[:] = old_qval[:]
                update = (reward + (gamma * maxQ))
                #the above is to form the target; we can use another more slowly updated Q network for this
                y[0][action] = update
                X_train.append(old_state.reshape(N_m, ))
                y_train.append(y.reshape(N_a, ))

            X_train = numpy.array(X_train)
            y_train = numpy.array(y_train)

            _, loss_value = sess.run([updateModel,loss], feed_dict={inputs: X_train, nextQ: y_train})
            #print loss_value

            eval_sum=0

            for eval_s in range(0,1000):
                _, Q_pred = sess.run([predict, Q_out], feed_dict={inputs: eval_set[eval_s].reshape(1, N_m)})
                eval_sum+=numpy.max(Q_pred)
            print eval_sum/1000
            eval_Q_sum[i] = eval_sum / 1000
            state = new_state

            if epsilon > 0.1:  # decrement epsilon over time
               epsilon -= (1 / ( epochs))



for eval_s in range(0,1000):
    _, Q_pred = sess.run([predict, Q_out], feed_dict={inputs: eval_set[eval_s].reshape(1, N_m)})
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
