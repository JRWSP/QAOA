# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:39:40 2020

@author: jiraw
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

np.random.seed(0)

data_size       = 10
n_clusters      = 2
cluster_center  = [[1, 1], [4, 4]]
pos, cluster    = make_blobs(n_samples=data_size, centers =cluster_center)

#plt.scatter(pos[:, 0], pos[:, 1])

k_means = KMeans(init = "k-means++", n_clusters = n_clusters, n_init = 1)

k_means.fit(pos)

k_means_labels          = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_

fig     = plt.figure(figsize=(6, 4))
colors  = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
ax      = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(cluster_center)), colors):
    my_members          = (k_means_labels == k)
    cluster_center_pos  = k_means_cluster_centers[k]
    ax.plot(pos[my_members, 0], pos[my_members, 1], 'x', markerfacecolor=col)
    ax.plot(cluster_center_pos[0], cluster_center_pos[1], 'o', markerfacecolor=col, markersize=6)

ax.set_xticks(())
ax.set_yticks(())
plt.show()
