# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:20:46 2020

@author: jiraw
"""

import numpy as np
import matplotlib.pyplot as plt


max_wC = {6: 52.419,
         7: 75.165,
         8: 83.683,
         9: 131.743,
         10: 132.297,
         11: 183.827,
         12: 240.005,
         13: 270.078}
max_uC = {6: 9.0,
         8: 16.0,
         10: 25.0,
         12: 36.0}
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
n=6

mean = np.array([])
std = np.array([])
for p in [1, 3, 5]:
#p=1
    temp = np.array([])
    #filename = "./wC/wC6nodes_p1_half_rough"
    filename = "./grid/grid_N"+str(n)+"_p"+str(p)+"_heuristic"
    data_temp = np.load(filename+".npy", allow_pickle=True)
    temp = np.array([data_temp[0][ii][0].fun for ii in range(len(data_temp[0]))])
    
    mean_temp = np.mean(temp)
    std_temp = np.std(temp)
    
    mean = np.append(mean, mean_temp/(-52.419))
    std = np.append(std, std_temp/(-52.419))
    
    plt.scatter(range(len(temp)), temp*100/-52.419)
plt.ylabel("Approx. ratio (%)")
plt.xticks([])
plt.title("Heuristic, 240initials")
plt.legend(["p=1", "p=3", "p=5"])
plt.savefig("./fig/heuristic", dpi=250)


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


"""
wC_mean = -np.mean(wC["nodes"+str(n)], axis=1)/(max_wC[n])
upp_err = [-np.min(a)/max_wC[n] for a in wC['nodes'+str(n)]]
upp_err = 100*(upp_err - wC_mean)
low_err = [-np.max(a)/max_wC[n] for a in wC['nodes'+str(n)]]
low_err = 100*(wC_mean - low_err)
plt.errorbar(range(1, 11), 100*wC_mean, yerr=[low_err, upp_err], color="C0", fmt='x')
plt.ylabel("r (%)")
plt.xlabel("p")
plt.legend()
plt.xticks(np.arange(1, 11))
"""
