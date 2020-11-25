import numpy as np
import networkx as nx
from qiskit import Aer
from sklearn.preprocessing import Normalizer

def FILE_HEURISTIC(nodes, layers, smple):
    if type(nodes) == type(layers) == type(smple):
        if METH == "COBYLA":
            filename = "./COBYLA/N"+str(nodes)+"/Heuristic_N" + str(nodes) + "_p" + str(layers) + "_sample" + str(smple) + "_COBYLA"
        elif METH == "Powell":
            filename = "./Heuristic/N"+str(nodes)+"/Heuristic_N" + str(nodes) + "_p" + str(layers) + "_sample" + str(smple)
        if LARGE_DISTANCE == True:
            filename = filename + "50dis"
        if NORM == True:
            filename = filename + "_norm"
        return filename
    else:
        raise Exception("Inputs should be integers.")


def FILE_RI(nodes, layers, smple):
    if type(nodes) == type(layers) == type(smple):
        if METH == "COBYLA":
        #filename = "./uC/uC" + str(nodes) + "nodes_p" + str(layers)
            filename = "./COBYLA/N"+str(nodes)+"/RI_N" + str(nodes) + "_p" + str(layers) + "_sample" + str(smple) + "_COBYLA"
        elif METH == "Powell":
            filename = "./RI/N"+str(nodes)+"/RI_N" + str(nodes) + "_p" + str(layers)+"_sample"+str(smple)
        if LARGE_DISTANCE == True:
            filename = filename + "_50dis"
        if NORM == True:
            filename = filename + "_norm"
        return filename
    else:
        raise Exception("Inputs should be integers.")

def GRID(nodes, layers, smple):
    if type(nodes) == type(layers) == type(smple):
        #filename = "./uC/uC" + str(nodes) + "nodes_p" + str(layers)
        filename = "./grid/Grid_N" + str(nodes) + "_p" + str(layers)+"_sample"+str(smple)
        if LARGE_DISTANCE == True:
            filename = filename + "50dis"
        if NORM == True:
            filename = filename + "_norm"
        return filename
    else:
        raise Exception("Inputs should be integers.")

def FIGNAME(nodes, smple, Strat):
    if type(nodes) == type(smple):
        #return "./fig/uC_" + str(nodes) +"_sample"+str(smple)
        if Strat == "_Heuristic":
            filename =  "./fig/N" + str(nodes) +"_sample" + str(smple) + Strat
        elif Strat == "_RI":
            filename =  "./fig/N" + str(nodes) +"_sample" + str(smple) + Strat
        else:
            raise Exception("Invalid Strat.")
        if METH == "COBYLA":
            filename = filename + "_COBYLA"
        if LARGE_DISTANCE == True:
            filename = filename + "_50dis"
        if NORM == True:
            filename = filename + "_norm"
        return filename
    else:
        raise Exception("Inputs should be integers.")

def SetConstrain(layer):
    #Define constrain for gradient-based optimizers:
    #construct the bounds in the form of constraints
    bnd = [bnds['beta'] for ii in range(layer)]
    for ii in range(layer):
        bnd.append(bnds['gamma'])
    cons = []
    for factor in range(len(bnd)):
        lower, upper = bnd[factor]
        l = {'type': 'ineq',
             'fun': lambda x, lb=lower, i=factor: x[i] - lb}
        u = {'type': 'ineq',
             'fun': lambda x, ub=upper, i=factor: ub - x[i]}
        cons.append(l)
        cons.append(u)
    return cons

N = 6
LAYER = 10
SMPLE = 3
LARGE_DISTANCE= False
NORM = False
METH = "Powell"
#METH = "COBYLA"
STRAT = "_Heuristic"
#STRAT = "_RI"

if LARGE_DISTANCE == True:
    data = np.load('./wC/wC'+str(N)+"nodes50dis.npy", allow_pickle=True)
else:
    data = np.load('./wC/'+str(N)+"nodes_10samples.npy", allow_pickle=True)
#
dist = data[SMPLE]['dist']
dist_norm = Normalizer().transform(dist)

G = nx.from_numpy_matrix(dist)
G_norm = nx.from_numpy_matrix(dist_norm)
#n = len(G.nodes())
V = np.arange(0, N, 1)
E = []
E_norm = []
for e in G.edges():
    E.append((e[0], e[1], G[e[0]][e[1]]['weight']))
    E_norm.append((e[0], e[1], G_norm[e[0]][e[1]]['weight']))

#Testing new method for renormalize matrix.
Edge_norm = np.transpose(E_norm)
w_min = np.min(Edge_norm[-1])
dist_normnew = dist_norm/w_min
G_normnew = nx.from_numpy_matrix(dist_normnew)

backend     = Aer.get_backend("statevector_simulator")
bnds = {'beta': (0*np.pi, 0.50*np.pi), 'gamma': (0*np.pi, 2.00*np.pi)}


if __name__ == "__main__":
    print("None")
