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

#grid6 = np.load("./grid/grid6nodes_p1statevector.npy", allow_pickle=True)
#grid10 = np.load("./grid/grid10nodes_p1statevector.npy", allow_pickle=True)
sample = 0
grid6 = np.load("./grid/Grid6Sample"+str(sample)+"statevector.npy", allow_pickle=True)

val = []
for ii in grid6:
    ii = list(ii.values())
    #beta.append(ii[0])
    #gamma.append(ii[1])
    val.append(ii[2])


val2 = np.array(val)
val2 = val2.reshape(100, 1000)
val2 = -1*val2
val2 = np.around(val2, 3)/52.419

Slice = 25
mean = np.mean(val2[Slice])
signal = val2[Slice] - mean
fourier = np.fft.fft(signal)
plt.plot(range(len(fourier)), fourier)

"""
plt.figure(figsize=(10,6))
plt.contourf(val2)
plt.xticks([0, 250, 500, 750, 1000], [r"$-1.0\pi$", r"$-0.5\pi$", r"$0$", r"$0.5\pi$", r"$1.0\pi$"])
plt.xlabel(r"$\gamma$")
plt.yticks([0, 25, 50, 75, 100], [r"$0$", r"$0.125\pi$", r"$0.250\pi$", r"$0.375\pi$", r"$0.500\pi$"])
plt.ylabel(r"$\beta$")
plt.colorbar()
plt.title("N = 6")
#plt.savefig("grid10", dpi=300)
"""
