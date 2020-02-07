# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:31:02 2020

@author: jiraw
"""

import networkx as nx
import numpy as np
import random
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import scipy.spatial.distance as dst

def ran_weight(weight):
    if weight == True:
        return format(random.random(), '.3f')
    else:
        return 1.0

def make_clus(n_sizes, n_clusters):
    pos, cluster    = make_blobs(n_samples=n_sizes, centers=[(-3.0, 0.0), (3.0 , 0.0)])
    plt.scatter(pos[:, 0], pos[:, 1])
    return pos, cluster

def gen_graph(n_node, filename, weight=False):
    G = nx.complete_graph(n_node)
    Edge = []
    for i in G:
        for j in G[i]:
            if j > i:
                tmp_w = ran_weight(weight)
                Edge.append((i, j, tmp_w ))
                G.add_weighted_edges_from([(i, j, tmp_w)])
            else:
                continue
        
    dist = nx.to_numpy_matrix(G)
    print(dist)
    """
    pos = nx.spring_layout(G)
    nx.draw_networkx(G)
    nx.draw_networkx_edge_labels(G, pos)
    """
    #np.savetxt(filename, dist, delimiter=',', fmt='%.3f')


#gen_graph(n_node=7, filename='dist.csv', weight=False)
data, ans = make_clus(20, 2)
dist = np.round(dst.squareform(dst.pdist(data)), 3)
np.savetxt('1_pos_data.csv', data, delimiter=',', fmt='%.3f')
np.savetxt('2_clus_number.csv', ans, delimiter=',', fmt='%.3f')
np.savetxt('3_dist.csv', dist, delimiter=',', fmt='%.3f')