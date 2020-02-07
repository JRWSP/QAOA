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
    # apply the Ising type gates with angle gamma along the edges in E
    for edge in E:
        k = edge[0]
        l = edge[1]
        QAOA.cu1(-2*gamma, k, l)
        QAOA.u1(gamma, k)
        QAOA.u1(gamma, l)
    # then apply the single qubit X - rotations with angle beta to all qubits
    QAOA.barrier()
    QAOA.rx(2*beta, range(len(V)))
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

Nfeval = 1

def optim(params):
    global Nfeval, cost, history
    
    beta, gamma = params[0], params[1]
    counts = Qiskit_QAOA(beta, gamma, V, E)
    avr_C       = 0
    for sample in list(counts.keys()):
        # use sampled bit string x to compute C(x)
        x         = [int(num) for num in list(sample)]
        tmp_eng   = cost_function_C(x,G)
        # compute the expectation value and energy distribution
        avr_C     = avr_C    + counts[sample]*tmp_eng
    M1_sampled   = avr_C/shots
    
    print('{0:4d}, {1:3.3f}, {2:3.3f}, {3:3.3f}'.format(Nfeval, beta, gamma, -M1_sampled))
    history = np.vstack((history, params))
    cost.append(-M1_sampled)
    Nfeval += 1
    return -M1_sampled

"""
# Generating the butterfly graph with 5 nodes 
n     = 5
V     = np.arange(0,n,1)
E     =[(0,1,1.0),(0,2,1.0),(1,2,1.0),(3,2,1.0),(3,4,1.0),(4,2,1.0)] 
"""
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
# Generate plot of the Graph
colors       = ['r' for node in G.nodes()]
default_axes = plt.axes(frameon=True)
pos          = nx.spring_layout(G)

nx.draw_networkx(G, node_color=colors, node_size=600, alpha=1, ax=default_axes, pos=pos)
"""
#Random initial parameters
rand_beta = np.random.uniform(low=-0.25*np.pi, high=0.25*np.pi)
rand_beta = np.round(rand_beta, 3)
rand_gamma = np.random.uniform(low=-0.50*np.pi, high=0.50*np.pi)
rand_gamma = np.round(rand_gamma, 3)

init_params = [rand_beta, rand_gamma]
history = np.array(init_params)
cost = []
print('{0:4s}, {1:9s}, {2:9s}, {3:9s}'.format('Iter', 'Beta', 'Gamma', 'Cost'))

bnds = ((-0.25*np.pi, 0.25*np.pi), (-.50*np.pi, 0.50*np.pi))
backend      = Aer.get_backend("qasm_simulator")
shots        = 10000


res = optimize.minimize(optim, init_params, 
                        method='Nelder-Mead',
                        bounds=bnds, 
                        options={'disp': True, 'xatol':1e-1, 'fatol':1.0})
"""
res = optimize.minimize(optim, init_params, 
                        method='SLSQP',
                        options={'disp': True, 'eps': 1e-03, 'ftol':1e-01}, 
                        bounds=bnds)
"""

if res.success:
    best_params = res.x
else:
    raise ValueError(res.message)

best_beta, best_gamma = best_params[0], best_params[1]
shots = 10000
result = Qiskit_QAOA(best_beta, best_gamma, V, E)

#plot_histogram(result,figsize = (8,6),bar_labels = False, title='Prob_distribution of the final state.')

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
print('The approximate solution is x* = %s with C(x*) = %d \n' % (max_C[0],max_C[1]))
#plot_histogram(hist,figsize = (8,6),bar_labels = False, title='Distribution of the cost function')

plt.figure()
for var in range(2):
    plt.plot(range(len(history[1:,var])), history[1:,var])

