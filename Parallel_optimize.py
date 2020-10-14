# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:28:48 2020

@author: jiraw
"""

import numpy as np
#import scipy.optimize as optimize
#import networkx as nx
from multiprocessing import Pool
from tqdm import tqdm, trange

#import matplotlib.pyplot as plt
#from   matplotlib import cm
#from   matplotlib.ticker import LinearLocator, FormatStrFormatter

#from qiskit import Aer, IBMQ
#from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
#from qiskit import quantum_info

#from qiskit.providers.ibmq      import least_busy
#from qiskit.tools.monitor       import job_monitor
#from qiskit.visualization import plot_histogram
import task
from config import *


if __name__ == "__main__":
    #n = 6

    n_cores = 24
    Iters = 24

    Result = []

    for layers in range(1,6):
        global bounds, p
        p = layers
        bounds = [ bnds['beta'] ]*p + [ bnds['gamma'] ] *p
        task.p_bounds(p = p, bounds = bounds)
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
        print("done p =", p)
    else:
        raise TypeError("Check func")


    filename = "./grid/grid_N"+str(n)+"_p"+str(p)+"_heuristic"
    np.save(filename, Result, allow_pickle=True)
    """
