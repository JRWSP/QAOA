# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:28:48 2020

@author: jiraw
"""

import numpy as np
import matplotlib.pyplot as plt
import config
import task
from qiskit import Aer, IBMQ
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit import quantum_info
from qiskit.visualization import plot_histogram
import networkx as nx

def get_uniform(*args, **kwargs):
    QAOA = QuantumCircuit(len(config.V), len(config.V))
    QAOA.h(range(len(config.V)))
    simulate     = execute(QAOA, backend=config.backend)
    QAOA_results = simulate.result()
    # Evaluate the data from the simulator
    statevector = QAOA_results.get_statevector(decimals=3)
    vec = quantum_info.Statevector(statevector)
    return vec.probabilities_dict(decimals=3)

def get_answer(*args, **kwargs):
    result = get_uniform()
    #plot_histogram(result,figsize = (8,6),bar_labels = False, title='Prob_distribution of the final state.')
    avr_C       = 0
    max_C       = [0,0]
    hist        = {}
    for k in range(len(config.G.edges())+1):
        hist[str(k)] = hist.get(str(k),0)
    for sample in list(result.keys()):
        # use sampled bit string x to compute C(x)
        x         = [int(num) for num in list(sample)]
        tmp_eng   = task.cost_function_C(x)
        # compute the expectation value and energy distribution
        avr_C     = avr_C    + result[sample]*tmp_eng
        hist[str(round(tmp_eng))] = hist.get(str(round(tmp_eng)),0) + result[sample]
        # save best bit string
        if( max_C[1] < tmp_eng):
            max_C[0] = sample
            max_C[1] = tmp_eng

    #M1_sampled   = avr_C
    print('\n --- SIMULATION RESULTS ---\n')
    #print(result)
    print("smple: ", config.SMPLE)
    print('The approximate solution is x* = %s with C(x*) = %.3f \n' % (max_C[0],max_C[1]))
    plot_histogram(hist,figsize = (8,6),bar_labels = False, title='Distribution of the cost function')

    global sol
    sol.append(-np.round(max_C[1], 3))

if __name__ == "__main__":
    sol = []
    get_answer()
        #plt.show()
    print(sol)
