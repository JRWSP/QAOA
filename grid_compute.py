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
    
    #QAOA.measure(range(len(V)),range(len(V)))
    ### draw the circuit for comparison
    #QAOA.draw(output='mpl')

    # run on local simulator
    simulate     = execute(QAOA, backend=backend)
    QAOA_results = simulate.result()
    # Evaluate the data from the simulator
    #counts = QAOA_results.get_counts()
    statevector = QAOA_results.get_statevector(decimals=3)
    vec = quantum_info.Statevector(statevector)
    return vec.probabilities_dict(decimals=3)



def rand_params(x, bnds, size):
    local_rand = np.random.RandomState(None)
    return np.round( local_rand.uniform(low=-bnds[x], high=bnds[x], size = size), 3 )


def optim(params):
    global history
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
    history = np.vstack((history, params))
    return -M1_sampled


if __name__ == "__main__":
    #Create graph from adjacency matrix
    n = 6
    #for n in tqdm([6]):
    for samp in tqdm(range(5)):
        data = np.load('./wC/'+str(n)+"nodes_10samples.npy", allow_pickle=True)
        dist = data[samp]['dist']
        
        G = nx.from_numpy_matrix(dist)
        n = len(G.nodes())
        V = np.arange(0, n, 1)
        E = []
        for e in G.edges():
            E.append((e[0], e[1], G[e[0]][e[1]]['weight']))
        
        p=1
        
        #Prepare Qiskit framework
        backend     = Aer.get_backend("statevector_simulator")
        #Sinigle core calculation
        Result = []
        for bbeta in np.linspace(0, 0.5*np.pi, 100):
            bbeta = np.round(bbeta, 3)
            print("\n n"+str(n)+"beta"+str(bbeta))
            for ggamma in np.linspace(-1.0*np.pi, 1.0*np.pi, 200):
                ggamma = np.round(ggamma, 3)
                init_params = list(np.concatenate(( [bbeta] , [ggamma])))
                state = Qiskit_QAOA( init_params[:p], init_params[p:], V, E)
                
                avr_C       = 0
                for sample in list(state.keys()):
                    # use sampled bit string x to compute C(x)
                    x         = [int(num) for num in list(sample)]
                    tmp_eng   = cost_function_C(x,G)
                    # compute the expectation value and energy distribution
                    avr_C     = avr_C    + state[sample]*tmp_eng
                M1_sampled   = -avr_C
                temp_res = {"beta": bbeta,"gamma": ggamma, "cost": np.round(M1_sampled, 3)}
                Result.append(temp_res)
                
        filename = "./grid/Grid"+str(n)+"Sample" +str(samp) +"Statevector"
        np.save(filename, Result, allow_pickle=True)
            