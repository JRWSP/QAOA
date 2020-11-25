# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:29:04 2020

@author: jiraw
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from config import *

sol6 = [-52.419, -35.872, -56.679, -51.35, -45.872, -55.649, -52.52, -57.38, -50.07, -51.043]
sol6_10dis = [-90.277, -89.716]

sol10 = [-132.297, -142.293, -155.565, -172.746, -174.125, -147.146, -176.785, -169.427, -142.972, -173.048]
sol10_10dis = [-243.761, -247.230]
if N == 6:
    if LARGE_DISTANCE == True:
        sol_min = sol6_10dis[SMPLE]
    else:
        sol_min = sol6[SMPLE]
elif N == 10:
    if LARGE_DISTANCE == True:
        sol_min = sol10_10dis[SMPLE]
    else:
        sol_min = sol10[SMPLE]
else:
    raise Exception("Check N")

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 15})

pp = input("Input p: ")
grid6 = np.load(GRID(N, int(pp), SMPLE)+".npy", allow_pickle=True)
#sample = 0

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
val2 = val2.reshape(400, 400)
val2 = val2/sol_min
val2 = np.around(val2, 3)
#val2 = val2.transpose()

beta = np.linspace(0.0*np.pi, 1.0*np.pi, 400)
gamma = np.linspace(0.0*np.pi, 2.0*np.pi, 400)
bbeta, ggamma = np.meshgrid(beta, gamma, indexing='ij')

#Find optimum point
betamax, gammamax = np.unravel_index(np.argmax(val2), np.shape(val2))

print("OpBeta: %.3f pi, OptGamma: %.3f pi" %(beta[betamax]/np.pi, gamma[gammamax]/np.pi) )
print(np.max(val2))
fig = plt.figure(figsize=(12, 10))

#surf = plt.contourf(bbeta, ggamma, val2, rstride=8, cstride=8, alpha=0.9, cmap=cm.coolwarm)

surf = plt.contourf(bbeta, ggamma, val2, zdir='x', rstride=8, cstride=8, alpha=0.9, cmap=cm.coolwarm)
plt.scatter(bbeta[betamax, gammamax], ggamma[betamax, gammamax])
"""
ax = fig.gca(projection='3d')
surf = ax.plot_surface(bbeta, ggamma, val2, rstride=8, cstride=8, alpha=0.9, cmap=cm.coolwarm)
cset = ax.contourf(bbeta, ggamma, val2, cmap=cm.coolwarm, zdir='x', offset=-0.5, alpha = 0.5)
cset = ax.contourf(bbeta, ggamma, val2, cmap=cm.coolwarm, zdir='y', offset=7.5, alpha = 0.5)
cset = ax.contourf(bbeta, ggamma, val2, cmap=cm.coolwarm, zdir='z', offset=0, alpha = 0.9)
#plt.plot(traj_gamma, traj_beta, "-o")
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$\gamma$")
ax.set_zlim(0.0, 1.0)
ax.zaxis.set_major_formatter('{x:.02f}')
"""
fig.colorbar(surf, shrink=0.5, aspect=5)


#plt.scatter(traj_gamma[1], traj_beta[1])
#plt.scatter(traj_gamma[-1], traj_beta[-1])

#for [a,b] in res[120:]:
#    plt.scatter(b, a)
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, 1.0*np.pi],
           [r"$0$", r"$0.25\pi$", r"$0.50\pi$", r"$0.75\pi$", r"$1.00\pi$"])
plt.xlabel(r"$\beta$")
plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
           [r"$0$", r"$0.5\pi$", r"$1.0\pi$", r"$1.5\pi$", r"$2.0\pi$"])
plt.ylabel(r"$\gamma$")



#plt.colorbar()
#plt.title("N = 6, P=2")
plt.show()
#plt.savefig("N6_sample0", dpi=250)

#-np.pi/2, -3*np.pi/8, -np.pi/4, -np.pi/8,
#r"$0.500\pi$", r"$0.375\pi$", r"$0.250\pi$", r"$0.125\pi$",
