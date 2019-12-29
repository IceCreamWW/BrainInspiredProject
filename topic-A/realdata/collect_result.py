import os
import numpy as np
from glob import iglob

def get_local_opts(loss_histories: list, opt_from_last_n=5):
    """ Get local optimum of experiments in loss_histories
    Arguments
    loss_histories:   list of histories of experiments
    opt_from_last_n:  select min value from last n loss values in history to be local optimum result of an experiment

    Return
    local optimums of all experiments
    """
    local_opts = []
    for history in loss_histories:
         local_opts.append(np.min(history[:-opt_from_last_n]))
    return np.array(local_opts)

def collect(n_samples):
    """Collect loss histories of a given sample size"""
    loss = []
    for filepath in iglob(f"./exp/{n_samples}/*/history.npy"):
        loss.append(np.loadtxt(filepath))
    return loss

if __name__=="__main__":
    for n_samples in [100, 500, 1000, 5000, 10000, 40000, 80000]:
        loss_histories = collect(n_samples)
        local_opts = get_local_opts(loss_histories)
        global_opt = np.min(local_opts)
        # smoothness = -np.mean(np.log(local_opts - global_opt))
        smoothness = (np.mean(local_opts - global_opt))
        print(f"n_sampels: {n_samples:7d}\t smoothness: {smoothness:.4f}")

