# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 21:25:19 2020

@author: OKCOM
"""
import numpy as np
import scipy.optimize as optimize
import networkx as nx

from qiskit import Aer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute


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
        #t_w = 1.0*(G[e1][e2]['weight']**2)/(2.0* sigma**2)
        #w = np.e**t_w
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
            #t_w = 1.0*(G[k][l]['weight']**2)/(2.0* sigma**2)
            #w = gamma[a]* np.e**t_w
            QAOA.cu1(-2*w, k, l)
            QAOA.u1(w, k)
            QAOA.u1(w, l)
            
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



def rand_params(x, bnds, size):
    local_rand = np.random.RandomState(None)
    return np.round( local_rand.uniform(low=bnds[x][0], high=bnds[x][1], size = size), 3 )


def optim(params):
    global history
    #beta, gamma = params[:p], params[p:]
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
    history = np.vstack((history, params))
    return -M1_sampled

def task(params):
    global history
    
    res = optimize.minimize(optim, params, method='Powell', bounds=bounds, options={'xtol':1e-3, 'ftol':1e-3})
    if res.success:
        print("\n ", params)
        return (res, history)
    else:
        raise ValueError(res.message)



n = 6
data = np.load('./wC/'+str(n)+"nodes_10samples.npy", allow_pickle=True)
dist = data[0]['dist']

G = nx.from_numpy_matrix(dist)
n = len(G.nodes())
V = np.arange(0, n, 1)
E = []
p = 1
bnds = {'beta': (0, 0.50*np.pi), 'gamma': (-0.25*np.pi, 0.25*np.pi)}
bounds = [ bnds['beta'] ]*p + [ bnds['gamma'] ] *p

backend      = Aer.get_backend("qasm_simulator")
shots        = 2**(n+2)

history = np.zeros(2*p)