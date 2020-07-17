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
        #t_w = -1.0*(G[e1][e2]['weight']**2)/(2.0* sigma**2)
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
            #t_w = -1.0*(G[k][l]['weight']**2)/(2.0* sigma**2)
            #w = gamma[a]* np.e**t_w
            """
            QAOA.cx(k, l)
            QAOA.rz(w, l)
            QAOA.cx(k, l)
            """
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
    #N = 8
    #Create graph from adjacency matrix
    sigma = 1.0
    for n in tqdm([6, 10]):
        #data = np.load('./Demo/w'+str(deg)+'R_8n.npy', allow_pickle=True)
        data = np.load('./wC/'+str(n)+"nodes_10samples.npy", allow_pickle=True)
        dist = data[0]['dist']
        
        G = nx.from_numpy_matrix(dist)
        n = len(G.nodes())
        V = np.arange(0, n, 1)
        E = []
        for e in G.edges():
            E.append((e[0], e[1], G[e[0]][e[1]]['weight']))
            #E.append((e[0], e[1], dist[e[0]][e[1]]))
        
        n_cores = 1
        #Iters = 48
        for p in range(1,2):
            
            #Construct initial boundaries as constraints
            bnds = {'beta': 0.50*np.pi, 'gamma': 1.00*np.pi}
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

            #Prepare Qiskit framework
            #backend      = Aer.get_backend("qasm_simulator")
            backend     = Aer.get_backend("statevector_simulator")
            #shots        = 2**(n+2)
            
            #Sinigle core calculation
            Result = []
            for bbeta in np.linspace(-0.5*np.pi, 0.5*np.pi, 100):
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
            
            #init_params = list(np.concatenate(( rand_params('beta', bnds, p) , rand_params('gamma', bnds, p))))
            
            #Parallel Optimization
            """
            Result = []
            history = np.zeros(2*p)
            def task(x):
                global history
                #Random initial parameters
                init_params = list(np.concatenate(( rand_params('beta', bnds, p) , rand_params('gamma', bnds, p))))
                if x%4 ==0:
                    print("\n n = " + str(n) + "p = " + str(p) + " Iteration " + str(x) + "/" + str(Iters-1) )
                res = optimize.minimize(optim, init_params, method='Powell', tol=1e-2)
                if res.success:
                    return (res, history)
                else:
                    raise ValueError(res.message)
                    
            with Pool(n_cores) as P:
                Sub_sample = P.map(task, range(Iters))
            Result.append(Sub_sample)
            #filename = "./QAOA_Data/var2.0/p" +str(p) +"shots" +str(shots)
            #filename = "./wC/wC_gaussian"+str(n)+"nodes_p" +str(p)
            """
            
            filename = "./grid/grid"+str(n)+"nodes_p" +str(p) +"statevector"
            np.save(filename, Result, allow_pickle=True)