#!/usr/bin/python
import numpy

import cplex
from cplex.exceptions import CplexError
import sys
import numpy
import itertools
from itertools import chain, combinations
import math
import copy

N_m=4
N_s=int(math.pow(2,N_m))
N_a=N_m+1
#all_actions=[0,1,2,3,4]
gamma=0.9

def powerset(iterable):
  xs = list(iterable)
  return chain.from_iterable( combinations(xs,n) for n in range(len(xs)+1) )

N_m_array=[]
for i in range(0,N_m):
    N_m_array.append(i)
all_basis=list(powerset(set(N_m_array)))
#print all_basis
def Rewards(x,a):
    reward=0
    #print x
    for i in range(0,N_m-1):
        reward=reward+x[i]
    #break symmetry
    reward=reward+2*x[N_m-1]
    return reward


def CPD(x_hat,x,a):
    p=1
    for i in range(0,len(x_hat)):
        if x_hat[i]==0:
           p=p*(1-probability_table(i,parent_indices(i),x,a))
        else:
           p=p*probability_table(i,parent_indices(i),x,a)
    return p

def States(index):
    all_states=list(itertools.product([0, 1], repeat=N_m))
    x=all_states[index]
    return x


def probability_table(i,parent_indices_x_hat,x,a):
    if a==i:
        prob=1
    else:
       if x[int(parent_indices_x_hat[0])]==0 and x[int(parent_indices_x_hat[1])]==0 :
          prob = 0.05

       if x[int(parent_indices_x_hat[0])]==0 and x[int(parent_indices_x_hat[1])]==1:
          prob=0.5

       if x[int(parent_indices_x_hat[0])]==1 and x[int(parent_indices_x_hat[1])]==0 :
          prob=0.09
       if x[int(parent_indices_x_hat[0])]==1 and x[int(parent_indices_x_hat[1])]==1:
          prob=0.9
    return prob

# this probability table is only for the two parent model


def parent_indices(i):
    p=numpy.zeros(2)
    if i==0:
       p[0]=0
       p[1]=N_m-1
    else:
       p[0]=i-1
       p[1]=i
    return p.astype(int)

num_basis=N_s-1
Italic_B = all_basis[0:N_m+1]
print Italic_B
def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)\

"""
alpha_sum=0
alpha=[]
for i in range(0,len(Italic_B)):
    l = len(Italic_B[i])
    numerator=math.pow(2,l-1)
    denominator=0
    for j in range(0,N_m+1):
        denominator=denominator+nCr(N_m,j)*math.pow(2,j-1)
    alpha.append(numerator/(float(denominator)))
    alpha_sum=alpha_sum+alpha[i]
"""


# define arbitrary alpha
#alpha=[0.11111111,0.22222222,0.22222222,0.44444444]
alpha=numpy.zeros((len(Italic_B)))

alpha[0]=1
#print alpha


"""
variables=[]
objective=[]
upper_bounds=[]
lower_bounds=[]
v_types=[]
#print len(alpha)
for t in range(0,len(Italic_B)):
    variables.append('w_' + str(t))
    #objective.append(alpha[t])
    upper_bounds.append(cplex.infinity)
    lower_bounds.append(-1 * cplex.infinity)
    sum_states=0
    for x in range(0,N_s):
        basis = Italic_B[t]
        if len(basis) == 0:
            temp = 1
        else:
            basis_sum = 0
            x_array = States(x)
            for k in range(0, len(basis)):
                basis_sum = basis_sum + x_array[basis[k]]
            temp = math.pow(-1, basis_sum)
        sum_states=sum_states+alpha[x]*temp
    print sum_states
    objective.append(sum_states)




prob = cplex.Cplex()
prob.objective.set_sense(prob.objective.sense.minimize)
prob.variables.add(obj = objective, ub = upper_bounds,lb=lower_bounds, names = variables)

"""

variables=[]
objective=[]
upper_bounds=[]
lower_bounds=[]
#Italic_B=Italic_B[0:2]
#print len(alpha)
for t in range(0,len(Italic_B)):
    variables.append('w_'+str(t))
    objective.append(alpha[t])
    upper_bounds.append(cplex.infinity)
    lower_bounds.append(-1*cplex.infinity)



prob = cplex.Cplex()
prob.objective.set_sense(prob.objective.sense.minimize)
prob.variables.add(obj = objective, ub = upper_bounds,lb=lower_bounds, names = variables)

prob.write("approx_primal.lp")


for a in range(0,N_a):
    for x in range(0,N_s):
        var=[]
        coef=[]
        rhs_term=Rewards(States(x),a)
        for i in range(0,len(Italic_B)):
            basis=Italic_B[i]
            if len(basis)== 0:
                coef_term=1
            else:
                basis_sum=0
                x_array=States(x)
                for k in range(0,len(basis)):
                    basis_sum=basis_sum+x_array[basis[k]]
                coef_term=math.pow(-1,basis_sum)

            var.append('w_' + str(i))

            #print coef_term
            sum_term=0
            for x_hat in range(0,N_s):
                if len(basis) == 0:
                    temp = 1
                else:
                    basis_sum = 0
                    x_array = States(x_hat)
                    for k in range(0, len(basis)):
                        basis_sum = basis_sum + x_array[basis[k]]
                    temp = math.pow(-1, basis_sum)

                sum_term=sum_term+temp*CPD(States(x_hat), States(x), a)

            sum_term=sum_term*gamma
            coef_term=coef_term-sum_term
            #print coef_term
            coef.append(coef_term)
        #print var
        #print coef
        #print rhs_term
        prob.linear_constraints.add(lin_expr=[cplex.SparsePair(var, coef)], senses=["G"], rhs=[rhs_term])

try:
    prob.write("approx_primal.lp")
    prob.solve()
except CplexError, exc:
    print exc

print ()
# solution.get_status() returns an integer code
print "Solution status = ", prob.solution.get_status(), ":",
# the following line prints the corresponding string
print(prob.solution.status[prob.solution.get_status()])
print("Solution value  = ", prob.solution.get_objective_value())

numcols = prob.variables.get_num()
numrows = prob.linear_constraints.get_num()

slack = prob.solution.get_linear_slacks()
x     = prob.solution.get_values()

for j in range(numrows):
    print("Row %d:  Slack = %10f" % (j, slack[j]))
for j in range(numcols):
    print("Column %d:  Value = %10f %s " % (j, x[j],variables[j]))