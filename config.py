import numpy as np
import networkx as nx
from qiskit import Aer

n = 6
data = np.load('./wC/'+str(n)+"nodes_10samples.npy", allow_pickle=True)
dist = data[0]['dist']

G = nx.from_numpy_matrix(dist)
n = len(G.nodes())
V = np.arange(0, n, 1)
E = []
for e in G.edges():
    E.append((e[0], e[1], G[e[0]][e[1]]['weight']))

#p = 1

backend     = Aer.get_backend("statevector_simulator")

bnds = {'beta': (-0.25*np.pi, 0.25*np.pi), 'gamma': (-0.50*np.pi, 0.50*np.pi)}
#bounds = [ bnds['beta'] ]*p + [ bnds['gamma'] ] *p
