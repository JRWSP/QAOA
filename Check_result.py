# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 23:11:50 2020

@author: jiraw
"""

import numpy as np
import matplotlib.pyplot as plt

Sol = '11100100'
Sol = np.array([int(n) for n in Sol])
pos = np.loadtxt('1_pos_data.csv', delimiter=',')
ans = np.loadtxt('2_clus_number.csv', delimiter=',')

n_clusters = 2

fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)
colors  = plt.cm.Spectral(np.linspace(0, 1, len(set(ans))))

sym = ['x', 'o']
for k, col in zip(range(n_clusters), colors):
    res_clas          = (ans == k)
    res_quan          = (Sol == k)
    #cluster_center_pos  = k_means_cluster_centers[k]
    ax0.plot(pos[res_clas, 0], pos[res_clas, 1], sym[k], markerfacecolor=col)
    plt.plot(pos[res_quan, 0], pos[res_quan, 1], sym[k], markerfacecolor=col)
    #ax.plot(cluster_center_pos[0], cluster_center_pos[1], 'o', markerfacecolor=col, markersize=6)


#plt.set_xticks(())
#plt.set_yticks(())
plt.show()