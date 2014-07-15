import numpy as np
import multiprocessing
import commands


def reset_kernel(nest,dic,n_threads=None,seed=None):
    print "\nResetting NEST kernel..."
    nest.ResetKernel()
    reset_seed(nest,n_threads,seed)
    if 'data_path' in dic.keys():
        print '\t*data path\t                : %s'%dic['data_path']
    nest.SetStatus([0],dic)

def reset_seed(nest,n_threads=None,seed=None):
    print '\t*number of cores available\t: %i'%multiprocessing.cpu_count()
    if n_threads==None:
        n_threads=multiprocessing.cpu_count()
        print '\t*local_num_threads\t        : %i'%n_threads
    else:
        print '\t*local_num_threads\t        : %i'%n_threads
    np.random.seed(seed)
    rndSeeds = np.random.randint(0,1000,n_threads+1)
    nest.SetStatus([0],{'local_num_threads'         : n_threads,
                        'rng_seeds'                 : rndSeeds[:n_threads].tolist(),
                        'grng_seed'                 : rndSeeds[n_threads]})
