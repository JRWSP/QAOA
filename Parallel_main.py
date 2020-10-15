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
    for p = range(2, 6):
        if p == 1:
            with Pool(n_cores) as P:
                Sub_sample = list(tqdm(P.imap(task.task_init, range(Iters)), total=Iters))
            Result.append(Sub_sample)
        elif p > 1:
            Sub_sample = task.task(layers=p)
            Result.append(Sub_sample)
            print("done p =", p)
        else:
            raise TypeError("Check func")


        filename = "./grid/grid_N"+str(n)+"_p"+str(p)+"_heuristic"
        np.save(filename, Result, allow_pickle=True)
