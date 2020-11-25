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
        params = [list(np.append(task.rand_params('beta', bnds, p), task.rand_params('gamma', bnds, p))) for ii in range(Iters)]
        with Pool(n_cores) as P:
            Sub_sample = list(tqdm(P.imap(task.task, params), total=len(params)))
        Result.append(Sub_sample)


        filename = FILE_RI(N, p, SMPLE)
        np.save(filename, Result, allow_pickle=True)
