# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:20:46 2020

@author: jiraw
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from config import FIGNAME, FILE_RI, FILE_HEURISTIC, smple, n

sol6 = [-52.419, -35.872, -56.679, -51.35, -45.872, -55.649, -52.52, -57.38, -50.07, -51.043]
sol6_10dis = [-90.277, -89.716]

sol10 = [-132.297, -142.293, -155.565, -172.746, -174.125, -147.146, -176.785, -169.427, -142.972, -173.048]
sol10_10dis = [-243.761, -247.230]
"""
hr = "./wC/wC6nodes_p1_half_rough"
hf = "./wC/wC6nodes_p1_half_fine"
fr = "./wC/wC6nodes_p1_full_rough"
ff = "./wC/wC6nodes_p1_full_fine"


ls = [hr, hf, fr, ff]
plt.figure()
for filename in ls:
    data_temp = np.load(filename+".npy", allow_pickle=True)
    temp = np.array([data_temp[0][ii][0].fun for ii in range(len(data_temp[0]))])
    x = range(len(temp))
    plt.scatter(x, -temp/52.419)
plt.legend(["half boundary + rough tolerance",
           "half boundary + fine tolerance",
           "full boundary + rough tolerance",
           "full boundary + fine tolerance"])
plt.ylabel("approx. ratio")
plt.savefig("comparing")


n = 10
mean = -np.mean(uC["nodes"+str(n)], axis=1)/(max_uC[n])
upp_err = [-np.min(a)/max_uC[n] for a in uC['nodes'+str(n)]]
upp_err = 100*(upp_err - mean)
low_err = [-np.max(a)/max_uC[n] for a in uC['nodes'+str(n)]]
low_err = 100*(mean - low_err)
plt.errorbar(range(1, 11), 100*mean, yerr=[low_err, upp_err], color="C0", fmt='o')
"""

last = 11
#for smple in range(1, 2):
Min_heu = []
mean_heu = []
std_heu = []

Min6 = []
mean = []
std = []
for p in range(1,last):
    mean_temp = 0
    std_temp = 0

    sol_min = sol6_10dis[smple]
    """
    file_heu = FILE_HEURISTIC(n, p, smple)
    data_temp = np.load(file_heu+".npy", allow_pickle=True)
    #data_temp = data_temp[::2]
    temp = np.array([data_temp[0][ii][0].fun for ii in range(len(data_temp[0]))])

    mean_temp = np.mean(temp)
    std_temp = np.std(temp)
    mean_heu = np.append(mean_heu, mean_temp/sol_min)
    std_heu = np.append(std_heu, std_temp/sol_min)
    Min_heu.append(np.min(temp)/sol_min)#sol10[smple])
    """

    file = FILE_HEURISTIC(n, p, smple)
    data_temp = np.load(file+".npy", allow_pickle=True)
    #data_temp = data_temp[::2]
    temp = np.array([data_temp[0][ii][0].fun for ii in range(len(data_temp[0]))])

    Op_Idx = np.argmin(temp)
    Op_fun = np.min(temp)

    mean_temp = np.mean(temp)
    std_temp = np.std(temp)
    mean = np.append(mean, mean_temp/sol_min)
    std = np.append(std, std_temp/sol_min)
    Min6.append(Op_fun/sol_min)#sol10[smple])

#Scatter avg and best costs with std error.
"""
plt.errorbar(np.arange(1, 11)+0.1, mean_heu,
             std_heu,
             fmt='o',
             capsize=5,
             color='b',
             ecolor='b',
             markerfacecolor='None',
             label="Heuristic search")
plt.scatter(np.arange(1, 11)+0.1, Min_heu,
            marker='x',
            color='b')
"""
plt.errorbar(np.arange(1, last), mean,
             std,
             fmt='o',
             capsize=5,
             color='k',
             ecolor='k',
             markerfacecolor='None',
             label="Heuristic search")
plt.scatter(np.arange(1, last), Min6,
            marker='x',
            color='k')
plt.ylabel("Approx. ratio")
plt.ylim([0.60, 1.0])
plt.xlabel("p")
plt.xticks(range(1,11))
#plt.title("Heuristic, 240initials")
plt.legend()
plt.grid(alpha=0.5)
#plt.savefig(FIGNAME(n, smple), dpi=250)
plt.show()




#Comparing two heuristic strategies
"""
#arr_max = [0.772, 0.837, 0.845, 0.851, 0.852]
#Heu_max = [0.777, 0.902, 0.923, 0.941, 0.954]
fig = plt.figure()
plt.scatter(range(1, 6), arr_max*100)
plt.scatter(range(1, 6), Heu_max*100)
plt.title("Max r from 240 Heuristic QAOA")
plt.xticks([1, 2, 3, 4, 5])
plt.xlabel("p")
plt.ylabel("Approx. ratio (%)")
plt.legend(["fix prior params", "optimize all params"])
plt.savefig("./fig/maxR", dpi=150)
"""
