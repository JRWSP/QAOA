import numpy as np
import networkx as nx
from qiskit import Aer
from sklearn.preprocessing import Normalizer

def FILE_HEURISTIC(nodes, layers, smple):
    if type(nodes) == type(layers) == type(smple):
        return "./Heuristic/N"+str(n)+"/Heuristic_N" + str(nodes) + "_p" + str(layers) + "_sample" + str(smple)+"10dis_norm"
    else:
        raise Exception("Inputs should be integers.")

def FILE_RI(nodes, layers, smple):
    if type(nodes) == type(layers) == type(smple):
        return "./RI"+str(n)+"/RI_N" + str(nodes) + "_p" + str(layers)+"_sample"+str(smple)
        #return "./uC/uC" + str(nodes) + "nodes_p" + str(layers)
    else:
        raise Exception("Inputs should be integers.")
def FIGNAME(nodes, smple):
    if type(nodes) == type(smple):
        #return "./fig/uC_" + str(nodes) +"_sample"+str(smple)
        return "./fig/Heuristic_N" + str(nodes) +"_sample" + str(smple) +"_dist_norm"
    else:
        raise Exception("Inputs should be integers.")

n = 6
layer = 10
smple = 0

#data = np.load('./wC/'+str(n)+"nodes_10samples.npy", allow_pickle=True)
data = np.load('./wC/wC'+str(n)+"nodes10dis.npy", allow_pickle=True)
dist = data[smple]['dist']
dist_norm = Normalizer().transform(dist)

G = nx.from_numpy_matrix(dist)
G_norm = nx.from_numpy_matrix(dist_norm)
n = len(G.nodes())
V = np.arange(0, n, 1)
E = []
E_norm = []
for e in G.edges():
    E.append((e[0], e[1], G[e[0]][e[1]]['weight']))
    #weight.append( np.exp(-G[e[0]][e[1]]['weight'])**2 / (2.0*0.5**2) )


backend     = Aer.get_backend("statevector_simulator")

bnds = {'beta': (-0.25*np.pi, 0.25*np.pi), 'gamma': (-0.50*np.pi, 0.50*np.pi)}

if __name__ == "__main__":
    #import matplotlib.pyplot as plt
    #plt.hist(weight)
    """
    pos = data[smple]['data']
    for point in pos:
        plt.scatter(point[0], point[1])
    """
    #plt.show()
