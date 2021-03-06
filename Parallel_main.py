# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:28:48 2020

@author: jiraw
"""

import numpy as np
#import scipy.optimize as optimize
#import networkx as nx
from multiprocessing import Pool
from tqdm import tqdm
import task
from config import *

if __name__ == "__main__":
    #n = 6
    n_cores = 24
    Iters = 240
    for p in range(1, 11):
        print("\n p: ", p)
        Result = []
        if p == 1:
            with Pool(n_cores) as P:
                Sub_sample = list(tqdm(P.imap(task.task_init, range(Iters)), total=Iters))
            Result.append(Sub_sample)
        elif p > 1:
            data_temp = np.load(FILE_HEURISTIC(N, p-1, SMPLE)+".npy", allow_pickle=True)
            params = np.array([data_temp[0][ii][0].x for ii in range(len(data_temp[0]))])
            #params = data_temp[0][np.argmin(temp)][0].x
            params = np.insert(params, p-1, [params[idx, p-2] for idx in range(len(params))], axis=1)
            params = np.insert(params, -1, [params[idx, -1] for idx in range(len(params))], axis=1)
            params = list(params)
            with Pool(n_cores) as P:
                Sub_sample = list(tqdm(P.imap(task.task, params), total=len(params)))
            #Sub_sample = task.task(layers = p)
            Result.append(Sub_sample)
            #print("done p =", p)
        else:
            raise TypeError("Check func")


        filename = FILE_HEURISTIC(N, p, SMPLE)
        np.save(filename, Result, allow_pickle=True)
