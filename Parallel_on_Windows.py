# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:28:48 2020

@author: jiraw
"""

import numpy as np
import scipy.optimize as optimize
import networkx as nx
from multiprocessing import Pool, cpu_count
from tqdm import tqdm, trange
from sklearn.preprocessing import Normalizer

import matplotlib.pyplot as plt
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter

from qiskit import Aer, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute

from qiskit.providers.ibmq      import least_busy
from qiskit.tools.monitor       import job_monitor
from qiskit.visualization import plot_histogram

import task

"""

def rand_params(x, bnds, size):
    local_rand = np.random.RandomState(None)
    return np.round( local_rand.uniform(low=bnds[x][0], high=bnds[x][1], size = size), 3 )
"""

def pick_initial():
    #Randomly choose initial parameters
    check = 1
    while check == 1:
        local_rand = np.random.RandomState(None)
        init_arg = [[ local_rand.randint(len(grid_beta)), local_rand.randint(len(grid_gamma)) ] for _ in range(Iters)]
        is_dup = {tuple(i) for i in init_arg}
        if is_dup == set(is_dup):
            check = 0
        else:
            continue
    return init_arg



if __name__ == "__main__":
    #Create graph from adjacency matrix
    n = 6

    n_cores = 24
    Iters = 48
    p = 1

    #Construct boundaries as constraints
    bnds = {'beta': (-0.25*np.pi, 0.25*np.pi), 'gamma': (-0.50*np.pi, 0.50*np.pi)}
    bounds = [ bnds['beta'] ]*p + [ bnds['gamma'] ] *p
    grid_beta = np.arange(bnds['beta'][0], bnds['beta'][1], 0.01, dtype=np.float)
    grid_gamma = np.arange(bnds['gamma'][0], bnds['gamma'][1], 0.01, dtype=np.float)

    args = pick_initial()
    init_params = [[ grid_beta[a], grid_gamma[b] ] for (a,b) in args]

    Result = []
    #test = task.task(init_params[0])
    with Pool(n_cores) as P:
        #Sub_sample = list(tqdm(P.imap(task.test, init_params), total = len(init_params)))
        Sub_sample = list(tqdm(P.imap(task.task, init_params), total = len(init_params)))
    Result.append(Sub_sample)

    filename = "./wC/wC6nodes_p1_half_fine"
    #filename = "./grid/initial_grid_p"+ str(p)
    np.save(filename, Result, allow_pickle=True)
