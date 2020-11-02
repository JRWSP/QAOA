# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:28:48 2020

@author: jiraw
"""

import numpy as np
import matplotlib.pyplot as plt
import config
import task
from qiskit.visualization import plot_histogram
import networkx as nx

def get_answer(best_params, *args, **kwargs):
    best_beta, best_gamma = best_params[:p], best_params[p:]
    result = task.Qiskit_QAOA(best_beta, best_gamma, config.V, config.E)
    #plot_histogram(result,figsize = (8,6),bar_labels = False, title='Prob_distribution of the final state.')

    avr_C       = 0
    max_C       = [0,0]
    hist        = {}
    for k in range(len(config.G.edges())+1):
        hist[str(k)] = hist.get(str(k),0)

    for sample in list(result.keys()):
        # use sampled bit string x to compute C(x)
        x         = [int(num) for num in list(sample)]
        tmp_eng   = task.cost_function_C(x,config.G)

        # compute the expectation value and energy distribution
        avr_C     = avr_C    + result[sample]*tmp_eng
        hist[str(round(tmp_eng))] = hist.get(str(round(tmp_eng)),0) + result[sample]

        # save best bit string
        if( max_C[1] < tmp_eng):
            max_C[0] = sample
            max_C[1] = tmp_eng

    M1_sampled   = avr_C

    """
    print('\n --- SIMULATION RESULTS ---\n')
    #print(result)
    print("smple: ", smple)
    print('The approximate solution is x* = %s with C(x*) = %.3f \n' % (max_C[0],max_C[1]))
    plot_histogram(hist,figsize = (8,6),bar_labels = False, title='Distribution of the cost function')
"""
    global sol6
    sol6.append(-np.round(max_C[1], 3))

if __name__ == "__main__":
    sol6 = []
    n = config.n
    p = 1
    for smple in range(1):
        #data = np.load('./wC/'+str(n)+"nodes_10samples.npy", allow_pickle=True)
        data = np.load('./wC/wC'+str(n)+"nodes10dis.npy", allow_pickle=True)
        dist = data[smple]['dist']

        config.G = nx.from_numpy_matrix(dist)
        config.n = len(config.G.nodes())
        config.V = np.arange(0, config.n, 1)
        config.E = []
        for e in config.G.edges():
            config.E.append((e[0], e[1], config.G[e[0]][e[1]]['weight']))

        #Load Data
        temp = np.array([])
        filename = "./Heuristic/Heuristic_N"+str(n)+"_p"+str(p)+"_sample"+str(smple)+"10dis"
        data_temp = np.load(filename+".npy", allow_pickle=True)
        temp = np.array([data_temp[0][ii][0].fun for ii in range(len(data_temp[0]))])
        #Get OptParamerters
        OptIdx = np.argmin(temp)
        OptParams = data_temp[0][OptIdx][0].x
        get_answer(OptParams)
        #plt.show()
    print(sol6)
