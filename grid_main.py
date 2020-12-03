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
import grid_compute
from config import *

if __name__ == "__main__":

    n_cores = 20
    p=LAYER

    if_save = 0

    if p == 1:
        beta = np.linspace(0.0*np.pi, 0.50*np.pi, 400)
        with Pool(n_cores) as P:
            Res = list(tqdm(P.imap(grid_compute.grid_compute, beta), total=len(beta)))
        if_save = 1

    elif p > 1:
        beta = np.linspace(0.0*np.pi, 0.50*np.pi, 400)
        with Pool(n_cores) as P:
            Res = list(tqdm(P.imap(grid_compute.grid_compute_highP, beta), total=len(beta)))
        if_save = 1

    else:
        print("Check optimum pre-initial beta and gamma.")

    if if_save == 1:
        filename = GRID(N, p, SMPLE)
        np.save(filename, Res, allow_pickle=True)
        print("\n Res is saved.")
    else:
        print("\n Res is not saved.")
