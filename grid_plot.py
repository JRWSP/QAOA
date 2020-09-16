# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:29:04 2020

@author: jiraw
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 15})

"""
graph = np.load("./wc/10nodes_10samples.npy", allow_pickle=True)
dist = graph[0]['dist']
hist_dist = []
for row in range(len(dist)):
    for column in range(row):
        hist_dist.append(dist[row][column])
        
hist_dist = np.array(hist_dist)
#plt.hist(hist_dist)

sigma = 3.0
hist_gaus = np.e**(1.0*(hist_dist**2)/(2.0* sigma**2))
plt.hist(hist_gaus)

plt.xlabel(r"Euclidian weight (N=6, $\sigma$ = 3.0)")
#plt.savefig("2_1")
"""
#grid6 = np.load("./grid/grid6nodes_p1statevector.npy", allow_pickle=True)
#sample = 0

res6 = np.load("./grid/initial_grid_p1.npy", allow_pickle=True)

bnds = {'beta': (0, 0.50*np.pi), 'gamma': (-0.50*np.pi, 0.50*np.pi)}
bounds = [ bnds['beta'] ] + [ bnds['gamma'] ]
grid_beta = np.arange(bnds['beta'][0], bnds['beta'][1], 0.01, dtype=np.float)
grid_gamma = np.arange(bnds['gamma'][0], bnds['gamma'][1], 0.01, dtype=np.float)

res = np.array(res6[0][0][0].x)
for ii in range(1, 240):
    res = np.vstack((res, res6[0][ii][0].x))
    
"""
val = []
for ii in grid6:
    ii = list(ii.values())
    #beta.append(ii[0])
    #gamma.append(ii[1])
    val.append(ii[2])


val2 = np.array(val)
val2 = val2.reshape(100, 200)
val2 = -1*val2#/52.419
val2 = np.around(val2, 3)
"""

plt.figure(figsize=(10,6))
plt.contourf(val2)
plt.xticks([0, 50, 100, 150, 200], [r"$-1.0\pi$", r"$-0.5\pi$", r"$0$", r"$0.5\pi$", r"$1.0\pi$"])
plt.xlabel(r"$\gamma$")
plt.yticks([0, 25, 50, 75, 100], [r"$0$", r"$0.125\pi$", r"$0.250\pi$", r"$0.375\pi$", r"$0.500\pi$"])
plt.ylabel(r"$\beta$")
plt.colorbar()
plt.title("N = 6, P=2")
plt.show()
#plt.savefig("grid6_p2"+str(sample), dpi=300)
