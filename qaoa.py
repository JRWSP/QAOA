# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:28:48 2020

@author: jiraw
"""

import numpy as np
import scipy.optimize as optimize
import networkx as nx
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt 
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter

from qiskit import Aer, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute

from qiskit.providers.ibmq      import least_busy
from qiskit.tools.monitor       import job_monitor
from qiskit.visualization import plot_histogram
from config import *

# Compute the value of the cost function
def cost_function_C(x,G):
    E = G.edges()
    if( len(x) != len(G.nodes())):
        return np.nan
    C = 0;
    for index in E:
        e1 = index[0]
        e2 = index[1]
        w      = G[e1][e2]['weight']
        C = C + w*x[e1]*(1-x[e2]) + w*x[e2]*(1-x[e1])
    return C

def Qiskit_QAOA(beta, gamma, V, E):
    # preapre the quantum and classical resisters
    QAOA = QuantumCircuit(len(V), len(V))
    # apply the layer of Hadamard gates to all qubits
    QAOA.h(range(len(V)))
    QAOA.barrier()
    
    for a in range(p):
        # apply the Ising type gates with angle gamma along the edges in E
        for edge in E:
            k = edge[0]
            l = edge[1]
            w = gamma[a]*G[k][l]['weight']
            QAOA.cu1(-2*w, k, l)
            QAOA.u1(w, k)
            QAOA.u1(w, l)
        # then apply the single qubit X - rotations with angle beta to all qubits
        QAOA.barrier()
        QAOA.rx(2*beta[a], range(len(V)))
        # Finally measure the result in the computational basis
        QAOA.barrier()
    
    QAOA.measure(range(len(V)),range(len(V)))
    ### draw the circuit for comparison
    #QAOA.draw(output='mpl')

    # run on local simulator
    simulate     = execute(QAOA, backend=backend, shots=shots)
    QAOA_results = simulate.result()
    # Evaluate the data from the simulator
    counts = QAOA_results.get_counts()
    return counts

def optim(params):
    global Nfeval, cost, history
    
    beta, gamma = params[:p], params[p:]
    counts = Qiskit_QAOA(beta, gamma, V, E)
    avr_C       = 0
    for sample in list(counts.keys()):
        # use sampled bit string x to compute C(x)
        x         = [int(num) for num in list(sample)]
        tmp_eng   = cost_function_C(x,G)
        # compute the expectation value and energy distribution
        avr_C     = avr_C    + counts[sample]*tmp_eng
    M1_sampled   = avr_C/shots
    if Nfeval % 10 == 0:
        print ("Nfeval: ", Nfeval)
        #print('{0:4d}, {1:3.3f}, {2:3.3f}, {3:3.3f}'.format(Nfeval, beta, gamma, -M1_sampled))
    history = np.vstack((history, params))
    cost.append(-M1_sampled)
    Nfeval += 1
    return -M1_sampled

def rand_params(x, bnds, size):
    return np.round( np.random.uniform(low=-bnds[x], high=bnds[x], size = size), 3 )
"""
# Generating the butterfly graph with 5 nodes 
n     = 5
V     = np.arange(0,n,1)
E     =[(0,1,1.0),(0,2,3.0),(1,2,3.0),(3,1,3.0),(3,4,2.0),(4,2,2.0)] 

#Create graph from adjacency matrix
dist = np.loadtxt('3_dist.csv', delimiter=',')
n = len(dist[0])
V = np.arange(0, n, 1)
E = []
for j in range(n):
    for k in range(1, n-j):
        E.append((j, j+k, dist[j, j+k]))

G     = nx.Graph()
G.add_nodes_from(V)
G.add_weighted_edges_from(E)
"""

"""
# Generate plot of the Graph
colors       = ['r' for node in G.nodes()]
default_axes = plt.axes(frameon=True)
pos          = nx.spring_layout(G)
nx.draw_networkx(G, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)
"""

p=1

#Construct boundaries as constraints
bnds = {'beta': 0.25*np.pi, 'gamma': 0.50*np.pi}
bounds = [ [ -bnds['beta'], bnds['beta'] ], 
           [ -bnds['gamma'], bnds['gamma'] ] ]
cons = []
for factor in range(len(bounds)):
    lower, upper = bounds[factor]
    l = {'type': 'ineq',
         'fun': lambda x, lb=lower, i=factor: x[i] - lb}
    u = {'type': 'ineq',
         'fun': lambda x, ub=upper, i=factor: ub - x[i]}
    for ii in range(p):
        cons.append(l)
        cons.append(u)

Itertion = 1
H = np.zeros(Itertion)
for ii in range(Itertion):
    print(ii)
    #Random initial parameters
    init_params = rand_params('beta', bnds, p)
    init_params = list(np.concatenate((init_params, rand_params('gamma', bnds, p))))
    history = np.array(init_params)
    cost = []
    #print('{0:4s}, {1:9s}, {2:9s}, {3:9s}'.format('Iter', 'Beta', 'Gamma', 'Cost'))
    
    #Optimize cost function
    Nfeval = 1
    backend      = Aer.get_backend("qasm_simulator")
    shots        = 100
    res = optimize.minimize(optim, init_params, method='COBYLA', constraints=cons, options={'disp': True})
    if res.success:
    	best_params = res.x
        #best_params = {'beta': res.x[0], 'gamma': res.x[1]}
    else:
        raise ValueError(res.message)
    """
    for line in range(p):
        plt.plot(range(len(history[:, line])), history[:,line], color='r')
    for line in range(p, 2*p):
        plt.plot(range(len(history[:, line])), history[:, line], color='b')
        """
    H[ii] = res.fun


#Get optimal solution
shots = 100
best_beta, best_gamma = best_params[:p], best_params[p:]
result = Qiskit_QAOA(best_beta, best_gamma, V, E)
plot_histogram(result,figsize = (8,6),bar_labels = False, title='Prob_distribution of the final state.')

avr_C       = 0
max_C       = [0,0]
hist        = {}
for k in range(len(G.edges())+1):
    hist[str(k)] = hist.get(str(k),0)
    
for sample in list(result.keys()):
    
    # use sampled bit string x to compute C(x)
    x         = [int(num) for num in list(sample)]
    tmp_eng   = cost_function_C(x,G)
    
    # compute the expectation value and energy distribution
    avr_C     = avr_C    + result[sample]*tmp_eng
    hist[str(round(tmp_eng))] = hist.get(str(round(tmp_eng)),0) + result[sample]
    
    # save best bit string
    if( max_C[1] < tmp_eng):
        max_C[0] = sample
        max_C[1] = tmp_eng
        
M1_sampled   = avr_C/shots

print('\n --- SIMULATION RESULTS ---\n')
print(res)
print('The approximate solution is x* = %s with C(x*) = %.3f \n' % (max_C[0],max_C[1]))
plot_histogram(hist,figsize = (8,6),bar_labels = False, title='Distribution of the cost function')

"""
plt.figure()
par = {0:"Beta", 1:"Gamma"}
for var in range(2):
    plt.plot(range(len(history[1:,var])), history[1:,var], label=par[var])
    plt.legend()
    plt.xlabel('Itertion')
    """
