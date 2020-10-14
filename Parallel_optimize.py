# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:28:48 2020

@author: jiraw
"""

import numpy as np
import scipy.optimize as optimize
import networkx as nx
from multiprocessing import Pool
from tqdm import tqdm, trange
from sklearn.preprocessing import Normalizer

import matplotlib.pyplot as plt
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter

from qiskit import Aer, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit import quantum_info

from qiskit.providers.ibmq      import least_busy
from qiskit.tools.monitor       import job_monitor
from qiskit.visualization import plot_histogram
import task

"""
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
        QAOA.rx(2*beta[a], range(len(V)))
        # Finally measure the result in the computational basis
        QAOA.barrier()

    # For Statevector_simulator
    # run on local simulator
    simulate     = execute(QAOA, backend=backend)
    QAOA_results = simulate.result()
    # Evaluate the data from the simulator
    #counts = QAOA_results.get_counts()
    statevector = QAOA_results.get_statevector(decimals=3)
    vec = quantum_info.Statevector(statevector)
    return vec.probabilities_dict(decimals=3)

    #For Qasm backend
    QAOA.measure(range(len(V)),range(len(V)))
    ### draw the circuit for comparison
    #QAOA.draw(output='mpl')

    # run on local simulator
    simulate     = execute(QAOA, backend=backend, shots=shots)
    QAOA_results = simulate.result()
    # Evaluate the data from the simulator
    counts = QAOA_results.get_counts()
    return counts
"""


"""
def rand_params(x, bnds, size):
    local_rand = np.random.RandomState(None)
    return np.round( local_rand.uniform(low=bnds[x][0], high=bnds[x][1], size = size), 3 )
"""

"""
def optim(params):
    global history
    beta, gamma = params[:p], params[p:]
    counts = Qiskit_QAOA(beta, gamma, V, E)
    avr_C       = 0
    for sample in list(state.keys()):
        # use sampled bit string x to compute C(x)
        x         = [int(num) for num in list(sample)]
        tmp_eng   = cost_function_C(x,G)
        # compute the expectation value and energy distribution
        avr_C     = avr_C + state[sample]*tmp_eng
    M1_sampled   = avr_C
"""

"""
    for sample in list(counts.keys()):
        # use sampled bit string x to compute C(x)
        x         = [int(num) for num in list(sample)]
        tmp_eng   = cost_function_C(x,G)
        # compute the expectation value and energy distribution
        avr_C     = avr_C    + counts[sample]*tmp_eng
    M1_sampled   = avr_C/shots
"""

"""
    history = np.vstack((history, params))
    return -M1_sampled
"""

if __name__ == "__main__":
    n = 6
    """
    #N = 8
    #Create graph from adjacency matrix

    #data = np.load('./Demo/w'+str(deg)+'R_8n.npy', allow_pickle=True)
    data = np.load('./wC/'+str(n)+"nodes_10samples.npy", allow_pickle=True)
    dist = data[0]['dist']

    G = nx.from_numpy_matrix(dist)
    #n = len(G.nodes())
    V = np.arange(0, n, 1)
    E = []
    for e in G.edges():
        E.append((e[0], e[1], G[e[0]][e[1]]['weight']))
        #E.append((e[0], e[1], dist[e[0]][e[1]]))
    """
    n_cores = 24
    Iters = 240
    for p in range(2,6):
        """
        #Construct boundaries as constraints
        bnds = {'beta': (-0.25*np.pi, 0.25*np.pi), 'gamma': (-0.50*np.pi, 0.50*np.pi)}
        bounds = [ bnds['beta'] ]*p + [ bnds['gamma'] ] *p

        backend     = Aer.get_backend("statevector_simulator")
        #backend      = Aer.get_backend("qasm_simulator")
        #shots        = 2**(n+2)
        """
        Result = []
        """
        history = np.zeros(2*p)
        def task(x):
            global history
            #Random initial parameters
            init_params = list(np.concatenate(( rand_params('beta', bnds, p) , rand_params('gamma', bnds, p))))
            if x%4 ==0:
                print("\n n = " + str(n) + "p = " + str(p) + " Iteration " + str(x) + "/" + str(Iters-1) )
            res = optimize.minimize(optim, init_params, method='Powell', bounds=bounds, options={'xtol':1e-4, 'ftol':1e-4})
            if res.success:
                return (res, history)
            else:
                raise ValueError(res.message)
        """
        if p == 1:
            with Pool(n_cores) as P:
                Sub_sample = list(tqdm(P.imap(task.task_init, range(Iters)), total=Iters))
            Result.append(Sub_sample)
        elif p > 1:
            #Get optimum parameters from a prior layer
            filename = "./grid/grid_N"+str(n)+"_p"+str(p-1)+"_heuristic"
            data_temp = np.load(filename+".npy", allow_pickle=True)
            temp = np.array([data_temp[0][ii][0].fun for ii in range(len(data_temp[0]))])
            params = data_temp[0][np.argmin(temp)][0].x
            params = np.insert(params, p-1, params[p-2])
            params = np.append(params, params[-1])
            Sub_sample = task.task(params)
            Result.append(Sub_sample)
        else:
            raise TypeError("Check func")


        filename = "./grid/grid_N"+str(n)+"_p"+str(p)+"_heuristic"
        np.save(filename, Result, allow_pickle=True)
