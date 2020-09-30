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

grid6 = np.load("./grid/Grid_p5.npy", allow_pickle=True)
#sample = 0
#arr_max = [0.772, 0.837, 0.845, 0.851, 0.852]
"""
#res6 = np.load("./grid/initial_grid_p1.npy", allow_pickle=True)
res = np.array(res6[0][0][0].x)
for ii in range(1, 240):
    res = np.vstack((res, res6[0][ii][0].x))

traj_beta = []
traj_gamma = []
for [a,b] in res6[0, 0, 1]:
    traj_beta.append(a)
    traj_gamma.append(b)
"""

val = []
for b in grid6:
    for g in b:
        val.append(list(g.values()))
"""
val = []
for ii in grid6:
    ii = list(ii.values())
    val.append(ii)
"""
val = np.array(val)
val2 = val[:,2]
val2 = val2.reshape(200, 200)
val2 = -1*val2/52.419
val2 = np.around(val2, 3)

ggamma = np.linspace(-np.pi, np.pi, 200)
bbeta = np.linspace(-0.5*np.pi, 0.5*np.pi, 200)

#Find optimum point
OpIdx = np.argmax(val2)
OptBeta = OpIdx // 200
OptGamma = OpIdx % 200
print("OpBeta: %.3f pi, OptGamma: %.3f pi" %(bbeta[OptBeta]/np.pi, ggamma[OptGamma]/np.pi) )

plt.figure(figsize=(10,6))
plt.contourf(ggamma, bbeta, val2)
#plt.plot(traj_gamma, traj_beta, "-o")

plt.scatter(ggamma[OptGamma], bbeta[OptBeta])

#plt.scatter(traj_gamma[1], traj_beta[1])
#plt.scatter(traj_gamma[-1], traj_beta[-1])

#for [a,b] in res[120:]:
#    plt.scatter(b, a)
"""
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
           [r"$-1.0\pi$", r"$-0.5\pi$", r"$0$", r"$0.5\pi$", r"$1.0\pi$"])
plt.xlabel(r"$\gamma$")
plt.yticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2],
           [r"$-0.500\pi$", r"$-0.250\pi$", r"$0$", r"$0.250\pi$", r"$0.500\pi$"])
plt.ylabel(r"$\beta$")
"""
plt.colorbar()
#plt.title("N = 6, P=2")
plt.show()
#plt.savefig("Checking2", dpi=300)

#-np.pi/2, -3*np.pi/8, -np.pi/4, -np.pi/8,
#r"$0.500\pi$", r"$0.375\pi$", r"$0.250\pi$", r"$0.125\pi$",