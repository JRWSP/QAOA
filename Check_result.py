# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 23:11:50 2020

@author: jiraw
"""

import numpy as np
import matplotlib.pyplot as plt

Sol = '10101000011101010100'
Sol = np.array([int(n) for n in Sol])
pos = np.loadtxt('1_pos_data.csv', delimiter=',')
ans = np.loadtxt('2_clus_number.csv', delimiter=',')
kmean = np.loadtxt('4_kmean_number.csv', delimiter=',')

n_clusters = 2

fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, constrained_layout=True)
colors  = plt.cm.Spectral(np.linspace(0, 1, len(set(ans))))

sym = ['x', '*', '+']
for k, col in zip(range(n_clusters), colors):
    res_clas          = (ans == k)
    res_quan          = (Sol == k)
    res_kmean            = (kmean == k)
    
    ax0.plot(pos[res_clas, 0], pos[res_clas, 1], sym[k], markerfacecolor=col)
    ax1.plot(pos[res_quan, 0], pos[res_quan, 1], sym[k], markerfacecolor=col)
    ax2.plot(pos[res_kmean, 0], pos[res_kmean, 1], sym[k], markerfacecolor=col)
    

ax0.title.set_text('Original')
ax1.title.set_text('QAOA')
ax2.title.set_text('K-Means')

#plt.set_xticks(())
#plt.set_yticks(())
plt.show()