import numpy as np
#import scipy.optimize as optimize
#import networkx as nx
from multiprocessing import Pool
from tqdm import tqdm, trange
import task
from config import n, smple

if __name__ == "__main__":
    #n = 6
    n_cores = 24
    Iters = 240
    for p in range(1, 11):
        It_p = [p] * Iters
        print("\n p: ", p)
        Result = []
        with Pool(n_cores) as P:
            Sub_sample = list(tqdm(P.imap(task.task_init, It_p), total=Iters))
        Result.append(Sub_sample)


        filename = "./RI/RI_N"+str(n)+"_p"+str(p)+"_sample" + str(smple)
        #filename = FILENAME(n, p, smple)
        np.save(filename, Result, allow_pickle=True)
