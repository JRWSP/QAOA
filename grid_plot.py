# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:29:04 2020

@author: jiraw
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 15})

pp = input("Input p: ")
grid6 = np.load("./grid/Grid_p"+str(pp)+".npy", allow_pickle=True)
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
val2 = val2.reshape(200, 200)
val2 = -1*val2/52.419
val2 = np.around(val2, 3)

beta = np.linspace(-0.5*np.pi, 0.5*np.pi, 200)
gamma = np.linspace(-np.pi, np.pi, 200)
bbeta, ggamma = np.meshgrid(beta, gamma)

#Find optimum point
betamax, gammamax = np.unravel_index(np.argmax(val2), val2.shape)

OpIdx = np.argmax(val2)
OptBeta = OpIdx // 200
OptGamma = OpIdx % 200
print("OpBeta: %.3f pi, OptGamma: %.3f pi" %(beta[OptBeta]/np.pi, gamma[OptGamma]/np.pi) )

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(ggamma, bbeta, val2, rstride=8, cstride=8, alpha=0.3, cmap=cm.coolwarm)
cset = ax.contourf(ggamma, bbeta, val2, cmap=cm.coolwarm, zdir='x', offset=-4)
cset = ax.contourf(ggamma, bbeta, val2, cmap=cm.coolwarm, zdir='y', offset=2)
cset = ax.contourf(ggamma, bbeta, val2, cmap=cm.coolwarm, zdir='z', offset=0)
#plt.plot(traj_gamma, traj_beta, "-o")
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"$\beta$")
ax.set_zlim(0.0, 1.0)
#ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.02f}')
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.plot(ggamma[OptBeta, OptGamma], bbeta[OptBeta, OptGamma], val2.max(), marker="o", ls="", c=cm.coolwarm(0.))

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
#plt.colorbar()
#plt.title("N = 6, P=2")
plt.show()
#plt.savefig("Checking2", dpi=300)

#-np.pi/2, -3*np.pi/8, -np.pi/4, -np.pi/8,
#r"$0.500\pi$", r"$0.375\pi$", r"$0.250\pi$", r"$0.125\pi$",
