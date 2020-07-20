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
grid6 = np.load("./grid/grid6nodes_p1statevector.npy", allow_pickle=True)
grid10 = np.load("./grid/grid10nodes_p1statevector.npy", allow_pickle=True)


val = []
for ii in grid10:
    ii = list(ii.values())
    #beta.append(ii[0])
    #gamma.append(ii[1])
    val.append(ii[2])


val2 = np.array(val)
val2 = val2.reshape(100, 200)
val2 = -1*val2/132.297
val2 = np.around(val2, 3)

plt.figure(figsize=(10,6))
plt.contourf(val2)
plt.xticks([0, 50, 100, 150, 200], [r"$-1.0\pi$", r"$-0.5\pi$", r"$0$", r"$0.5\pi$", r"$1.0\pi$"])
plt.xlabel(r"$\gamma$")
plt.yticks([0, 25, 50, 75, 100], [r"$-0.5\pi$", r"$-0.25\pi$", r"$0$", r"$0.25\pi$", r"$0.5\pi$"])
plt.ylabel(r"$\beta$")
plt.colorbar()
plt.title("N = 10")
plt.savefig("grid10", dpi=300)

