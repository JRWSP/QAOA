import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from config import *
from task import Qiskit_QAOA, cost_function_C
import itertools

#Don't foget to change p in task.py
p = LAYER

filename = FILE_HEURISTIC(N, p, SMPLE)
data_temp = np.load(filename+".npy", allow_pickle=True)
temp = np.array([data_temp[0][ii][0].fun for ii in range(len(data_temp[0]))])
Op_Idx = np.argmin(temp)
Op_fun = np.min(temp)
Op_x = data_temp[0][Op_Idx][0].x
#Op_x = data_temp[0][Op_Idx][1][-1]

plt.scatter(range(p), Op_x[:p], label="beta")
plt.scatter(range(p), Op_x[p:]*w_max, label="gamma")
plt.legend()

beta, gamma = Op_x[:p], Op_x[p:]
result = Qiskit_QAOA(beta, gamma, norm=NORM)
#result = dict(itertools.islice(result.items(), 32))
title = "N="+str(N)+", p="+str(p)

#fig1 = plot_histogram(result, figsize = (12,8),bar_labels = True, title=title)
plt.xlabel("Bitstring states")
plt.xticks([])

avr_C       = 0
max_C       = [0,0]
hist        = {}
#For What?
"""
for k in range(len(config.G.edges())+1):
    hist[str(k)] = hist.get(str(k),0)
"""
for sample in list(result.keys()):
    # use sampled bit string x to compute C(x)
    x         = [int(num) for num in list(sample)]
    x         = list(np.flip(x))
    tmp_eng   = cost_function_C(x)

    # compute the expectation value and energy distribution
    avr_C     = avr_C    + result[sample]*tmp_eng
    hist[int(round(tmp_eng))] = hist.get(int(round(tmp_eng)),0) + result[sample]

    # save best bit string
    if( max_C[1] < tmp_eng):
        max_C[0] = sample
        max_C[1] = tmp_eng
print('\n --- SIMULATION RESULTS ---\n')
print('The sampled mean value is M1_sampled = %.02f, Op_fun = %.2f \n' % (avr_C, Op_fun))
print('The approximate solution is x* = %s with C(x*) = %d \n' % (max_C[0],max_C[1]))
print('The cost function is distributed as: \n')
#fig2 = plot_histogram(hist,figsize = (12,8),bar_labels = False, title=title)

sol6 = [-52.419, -35.872, -56.679, -51.35, -45.872, -55.649, -52.52, -57.38, -50.07, -51.043]
sol6_10dis = [-90.277, -89.716]

sol10 = [-132.297, -142.293, -155.565, -172.746, -174.125, -147.146, -176.785, -169.427, -142.972, -173.048]
sol10_10dis = [-243.761, -247.230]

if N == 6:
    if LARGE_DISTANCE == True:
        sol_min = sol6_10dis[SMPLE]
    else:
        sol_min = sol6[SMPLE]
elif N == 10:
    if LARGE_DISTANCE == True:
        sol_min = sol10_10dis[SMPLE]
    else:
        sol_min = sol10[SMPLE]
else:
    raise Exception("Check N")

r = np.round(Op_fun*100/sol_min)

plt.show()
#fig1.savefig("./fig/N"+str(n)+"_p"+str(p)+"_Heuristic_prob.png")
#fig2.savefig("./fig/N"+str(n)+"_p"+str(p)+"_Heuristic_cost_"+str(r)+".png")
