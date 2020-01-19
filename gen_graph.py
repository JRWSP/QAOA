# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:31:02 2020

@author: jiraw
"""

import networkx as nx
import numpy as np
import random
def ran_weight():
    return format(random.random(), '.3f')

n_node= 5
G = nx.complete_graph(n_node)
Edge = []
for i in G:
    for j in G[i]:
        if j > i:
            tmp_w = ran_weight()
            Edge.append((i, j, tmp_w ))
            G.add_weighted_edges_from([(i, j, tmp_w)])
        else:
            continue
    
dist = nx.to_numpy_matrix(G)
"""
pos = nx.spring_layout(G)
nx.draw_networkx(G)
nx.draw_networkx_edge_labels(G, pos)
"""