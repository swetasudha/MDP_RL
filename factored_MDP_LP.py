#!/usr/bin/python
import numpy
import copy
import cplex
from cplex.exceptions import CplexError
import sys
import numpy
import itertools
from itertools import chain, combinations
import random
import math
import pickle
import timeit

N_m=12
N_a=N_m+1
N_s=numpy.power(2,N_m)
gamma=0.9
Ca=0
def powerset(iterable):
  xs = list(iterable)
  return chain.from_iterable( combinations(xs,n) for n in range(len(xs)+1) )

N_m_array=[]
for i in range(0,N_m):
    N_m_array.append(i)

def factored_rewards(j,a):
    if a==1:
       if j==N_m-1:
          reward= 2
       else:
          reward= 1
    else:
       reward=0
    return reward
def CPD(i,x_hat_i,y_p,a):
    if x_hat_i==0:
       p=(1-probability_table(i,y_p,a))
    else:
       p=probability_table(i,y_p,a)
    return p

def probability_table(i,y_p,a):

    if a==i:
       prob=1
    else:
       if y_p[0]==0 and y_p[1]==0:
          prob=0.05
       if y_p[0]==0 and y_p[1]==1:
          prob=0.5
       if y_p[0]==1 and y_p[1]==0:
          prob=0.09
       if y_p[0]==1 and y_p[1]==1:
          prob=0.9
    return prob

def parent_indices(i):
    p=numpy.zeros(2,dtype=int)
    if i==0:
       p[0]=0
       p[1]=N_m-1
    else:
       p[0]=i-1
       p[1]=i
    return p


def precompute_prob(Z_value,a):
    p_vec=numpy.zeros((N_m,))
    for i in range(0,N_m):
        p_vec[i]=p_func(i, Z_value, a)
    return p_vec

def p_func(i, Z_value, a):
    parents = parent_indices(i)
    values = []
    for j in range(0, len(parents)):
        values.append(Z_value[int(parents[j])])
        if Z_value[int(parents[j])]==2:
            return 0
    # return 1*probability_table(i,values,a)
    return 1 - 2 * probability_table(i, values, a)


def c_func(S_i, Z_value, a):
    g = 1
    basis_sum = 0
    for i in range(0, N_m):
        if S_i[i] == 1:
            basis_sum = basis_sum + Z_value[i]
            # if Z_value[i]==1:
            #   basis_sum=1
            # print p_func(i,Z_value,a)
            g = g * p_func(i, Z_value, a)

    # h=basis_sum
    h = int(math.pow(-1, basis_sum))

    # print basis_sum
    return gamma * g - h

alpha = [1, 0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0]


def Master_problem(Italic_B,current_basis_set,all_actions):
    #print current_basis_set
    # add objective and variables in cplex
    #print Italic_B
    variables = []
    objective = []
    upper_bounds = []
    lower_bounds = []
    v_types = []
    # print len(alpha)
    for t in range(0, len(Italic_B)):
        variables.append('w_' + str(t)+"_a")
        if t==0:
           objective.append(1)
        else:
           objective.append(0)
        upper_bounds.append(cplex.infinity)
        #lower_bounds.append(-1 * cplex.infinity)
        lower_bounds.append(0)
        variables.append('w_' + str(t) + "_b")
        if t==0:
           objective.append(-1)
        else:
           objective.append(0)
        upper_bounds.append(cplex.infinity)
        #lower_bounds.append(-1 * cplex.infinity)
        lower_bounds.append(0)



    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.minimize)
    prob.variables.add(obj=objective, ub=upper_bounds, lb=lower_bounds, names=variables)
    #prob.write("exact_primal.lp")


    constraint_index=[]
    constraint_index_count=0
    constraint_assignments=[]
    constraint_action=[]
    for actions in range(0, len(all_actions)):
        basis_set_remaining = []
        for b in range(1, len(Italic_B)):
            basis_set_remaining.append(b)  # 1 to N_s-1, this keeps track of the set of basis
        basis_set_remaining = numpy.flipud(numpy.arange(1, len(Italic_B), dtype=numpy.int))
        #print basis_set_remaining
        # this part is for LP
        # basis_set_remaining=[3,2,1]
        # functions still not used in any elimination
        elim_order = []
        #for b in range(1, N_m):
        #    elim_order.append(b)  # 1 to N_s-1, this keeps track of the set of basis
        elim_order = numpy.flipud(numpy.arange(1, N_m, dtype=numpy.int))
        # elim_order = [1, 0]
        old_scope = []  # this keeps track of the scope of the last introduced function
        for e_j in range(0, N_m-1):
            # print "new elimination \n"
            # t is the variable being eliminated
            t = elim_order[e_j]
            other_variable_set = []  # this set stores all variables that are involved while eliminating t
            basis_involved = []  # this keeps track of the basis functions involved while eliminating t
            for i in range(0, len(basis_set_remaining)):
                # S_i is the basis in binary format, binary vector of length N_m
                S_i = numpy.zeros((N_m, 1))
                basis_index = basis_set_remaining[i]
                B_i = Italic_B[basis_index]  # B_i is the actual basis
                parents=numpy.zeros((N_m, 1))
                for j in range(0, len(B_i)):
                    S_i[B_i[j]] = 1
                # parents is the binary vector for parents
                    p_basis=parent_indices(B_i[j])
                    for pp in range(0,len(p_basis)):
                        parents[int(p_basis[pp])]=1



                Z = parents + S_i

                if Z[t] != 0:
                    basis_involved.append(basis_index)
                # print basis_involved


                for w in range(0, N_m - e_j):  # out of remaining variables
                    if Z[w] != 0 and Z[t] != 0:
                        other_variable_set.append(w)

                # expand scope to include that of recently introduced eliminated variable function u_{old_scope}
                if len(old_scope) != 0:
                    other_variable_set = list(set(other_variable_set) | set(old_scope))
            # update basis remaining
            basis_set_remaining = list(set(basis_set_remaining) - set(basis_involved))

            other_variable_set = list(set(other_variable_set))

            external_variables = copy.deepcopy(other_variable_set)
            external_variables.remove(t)
            # the above are the set of variables except the one being eliminated
            # all possible assignments to all variables involved in eliminating t
            assignments = list(itertools.product([0, 1], repeat=len(other_variable_set)))
            for v in range(0, len(assignments)):
                # print len(assignments)
                var = []
                coef = []
                objective = []
                upper_bounds = []
                lower_bounds = []
                v_types = []
                current_assignment = assignments[v]
                name = "u_a" + str(all_actions[actions] + 1) + "_e" + str(N_m - t)
                # Z_value is a vector of length N_m, where elements are the assignments to values
                #  involved in eliminating t
                Z_value = numpy.zeros((N_m, 1))
                # index element is the position at which a variable appears (if at all) in other_variable_set
                for w in range(0, N_m):
                    try:
                        index_element = other_variable_set.index(w)
                    except ValueError:
                        index_element = -1
                    if index_element >= 0:
                        Z_value[w] = current_assignment[index_element]
                    else:
                        Z_value[w] = 2

                for w in range(0, len(external_variables)):
                    # find the position of a variable in external_variables, in the assignment vector
                    try:
                        index_element = other_variable_set.index(external_variables[w])
                    except ValueError:
                        index_element = -1

                    name = name + "_x" + str(external_variables[w] + 1) + "_" + str(current_assignment[index_element])
                var.append(name)
                coef.append(1)

                objective.append(0)
                upper_bounds.append(cplex.infinity)
                lower_bounds.append(-1 * cplex.infinity)

                # adding the u variables with scope =external variables, the functions that appear after eliminating t
                prob.variables.add(obj=objective, ub=upper_bounds, lb=lower_bounds, names=var)
                # defining rewards
                if Z_value[t] == 1:
                    if t == N_m - 1:
                        u_i_R = 2
                    else:
                        u_i_R = 1
                else:
                    u_i_R = 0


                for w in range(0, len(basis_involved)):
                    # adding the u variables corresponding to basis involved
                    var_1 = []
                    coef_1 = []
                    objective_1 = []
                    upper_bounds_1 = []
                    lower_bounds_1 = []
                    v_types_1 = []
                    name = "u_a" + str(all_actions[actions] + 1) + "_c" + str(basis_involved[w])
                    parents = []
                    B_i = Italic_B[basis_involved[w]]
                    try:
                        index_element = current_basis_set.index(B_i)
                    except ValueError:
                        index_element = -1
                    if index_element<0:
                        continue
                    # print B_i
                    for j in range(0, len(B_i)):
                        parents_j = parent_indices(B_i[j])
                        for jj in range(0, len(parents_j)):
                            parents.append(int(parents_j[jj]))
                    parents = list(set(parents))

                    for ww in range(0, len(parents)):
                        index_element = other_variable_set.index(parents[ww])
                        name = name + "_x" + str(parents[ww] + 1) + "_" + str(current_assignment[index_element])
                    var.append("w_"+str(basis_involved[w])+"_a")
                    S_i = numpy.zeros((N_m, 1))
                    B_i = Italic_B[basis_involved[w]]
                    for j in range(0, len(B_i)):
                        S_i[B_i[j]] = 1
                    coef.append(-1*(c_func(S_i, Z_value, all_actions[actions])))
                    var.append("w_"+str(basis_involved[w])+"_b")
                    coef.append(c_func(S_i, Z_value, all_actions[actions]))
                    if not (name in prob.variables.get_names()):  # add the u variables corresponding to basis only once
                        # while these will be used twice corresponding to 0 and 1 assignment of t
                        var_1.append(name)
                        coef_1.append(1)
                        objective_1.append(0)
                        upper_bounds_1.append(cplex.infinity)
                        lower_bounds_1.append(-1 * cplex.infinity)

                        #prob.variables.add(obj=objective_1, ub=upper_bounds_1, lb=lower_bounds_1, names=var_1)

                        S_i = numpy.zeros((N_m, 1))
                        B_i = Italic_B[basis_involved[w]]
                        for j in range(0, len(B_i)):
                            S_i[B_i[j]] = 1
                        var_1.append("w_" + str(basis_involved[w]))
                        coef_1.append(-1 * (c_func(S_i, Z_value, all_actions[actions])))

                        # add equality definition for the u basis variables
                        #prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=var_1, val=coef_1)], senses=["E"],
                                                    #rhs=[0])
                if len(old_scope) != 0:
                    name = "u_a" + str(all_actions[actions] + 1) + "_e" + str(e_j)
                    for w in range(0, len(old_scope)):
                        index_element = other_variable_set.index(old_scope[w])
                        # this must be already in other_variable_set since we always add old_scope to other_variable_set
                        name = name + "_x" + str(old_scope[w] + 1) + "_" + str(current_assignment[index_element])
                    var.append(name)
                    coef.append(-1)
                # add the full constraint for the current assignment
                constraint_assignments.append(Z_value)
                constraint_index.append(constraint_index_count)
                constraint_action.append(all_actions[actions])
                prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=var, val=coef)], senses=["G"], rhs=[u_i_R])
                #print constraint_index_count
                constraint_index_count=constraint_index_count+1
            # update old_scope
            old_scope = external_variables
        # add the final u variable u_N_m
        var = []
        coef = []
        objective = []
        upper_bounds = []
        lower_bounds = []

        name = "u_a" + str(all_actions[actions] + 1) + "_e" + str(N_m)
        var.append(name)
        coef.append(1)
        objective.append(0)
        upper_bounds.append(cplex.infinity)
        lower_bounds.append(-1 * cplex.infinity)

        prob.variables.add(obj=objective, ub=upper_bounds, lb=lower_bounds, names=var)

        # add the two constraints corresponding to the final variable
        for w in range(0, 2):
            var = []
            coef = []
            name = "u_a" + str(all_actions[actions] + 1) + "_e" + str(N_m)
            var.append(name)
            coef.append(1)
            name = "u_a" + str(all_actions[actions] + 1) + "_e" + str(N_m - 1)
            name = name + "_x" + str(old_scope[0] + 1) + "_" + str(w)
            var.append(name)
            coef.append(-1)
            u_i_R = w
            prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=var, val=coef)], senses=["G"], rhs=[u_i_R])
            constraint_index_count = constraint_index_count + 1

        # add the final constraint
        var = []
        coef = []
        var.append("w_0_a")
        coef.append(1-gamma)
        var.append("w_0_b")
        coef.append(-1*(1 - gamma))
        name = "u_a" + str(all_actions[actions] + 1) + "_e" + str(N_m)
        var.append(name)
        coef.append(-1)
        if all_actions[actions]==N_m:
           prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=var, val=coef)], senses=["G"], rhs=[0])
        else:
           prob.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=var, val=coef)], senses=["G"], rhs=[-1*Ca])
        constraint_index_count = constraint_index_count + 1
    #prob.write("exact_primal.lp")
    prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)
    try:
        prob.solve()
    except CplexError, exc:
        print exc
    """
    print ()
    # solution.get_status() returns an integer code
    print "Solution status = ", prob.solution.get_status(), ":",
    # the following line prints the corresponding string
    print(prob.solution.status[prob.solution.get_status()])
    print("Solution value  = ", prob.solution.get_objective_value())
    """
    solution_obj=prob.solution.get_objective_value()
    print solution_obj

    numcols = prob.variables.get_num()
    numrows = prob.linear_constraints.get_num()

    slack = prob.solution.get_linear_slacks()
    x = prob.solution.get_values()

    # print numcols
    variables = prob.variables.get_names()
    #for j in range(numcols):
    #    print("Column %d:  Value = %10f %s " % (j, x[j], variables[j]))
    basis_weights=[]
    for j in range(0,len(Italic_B)):
        weight_diff=x[2*j]-x[2*j+1]
        if weight_diff!=0:
            basis_weights.append([Italic_B[j],weight_diff])


    return prob.solution.get_dual_values(),constraint_assignments,constraint_index,constraint_action,basis_weights,solution_obj


current_basis_set = [()]
state_variables = numpy.arange(0, N_m)
for subset in itertools.combinations(state_variables, 1):
    current_basis_set.append(list(subset))
Italic_B=copy.deepcopy(current_basis_set)
print Italic_B
all_actions=[]
for i in range(0,N_a):
    all_actions.append(i)
Master_problem(Italic_B,Italic_B,all_actions)