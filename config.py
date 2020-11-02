import numpy as np
import networkx as nx
from qiskit import Aer

n = 6
smple = 0

#data = np.load('./wC/'+str(n)+"nodes_10samples.npy", allow_pickle=True)
data = np.load('./wC/wC'+str(n)+"nodes10dis.npy", allow_pickle=True)
dist = data[smple]['dist']

G = nx.from_numpy_matrix(dist)
n = len(G.nodes())
V = np.arange(0, n, 1)
E = []
weight = []
for e in G.edges():
    E.append((e[0], e[1], G[e[0]][e[1]]['weight']))
    weight.append( np.exp(-G[e[0]][e[1]]['weight'])**2 / (2.0*0.5**2) )


backend     = Aer.get_backend("statevector_simulator")

bnds = {'beta': (-0.25*np.pi, 0.25*np.pi), 'gamma': (-0.50*np.pi, 0.50*np.pi)}

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.hist(weight)
    """
    pos = data[smple]['data']
    for point in pos:
        plt.scatter(point[0], point[1])
    """
    plt.show()
