import sponet
import numpy as np
import sponet
from sponet.network_generator import ErdosRenyiGenerator
from sponet.collective_variables import OpinionShares
from sponet import sample_many_runs, CNVM, CNVMParameters, sample_cle
import scipy as sp
import matplotlib.pyplot as plt
from utils.network_utils import *
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")



with open("paths.json") as file:
    data = json.load(file)
data_path = data["data_path"]


def compare_wasserstein(
        params,
        x_init_network,
        n_runs,
        t_max,
        save_resolution = 2,
        simulation_resolution_sde = 20,
        verbose = False
):

    n_states = params.num_opinions
    n_nodes = params.num_agents

    cv = OpinionShares(n_states, normalize=True)

    x_init_shares = cv(np.array([x_init_network]))


    if verbose: print("Starting SSA simulation...")
    t_ref,x_ref = sample_many_runs(
        params,
        np.array([x_init_network]),
        t_max,
        save_resolution*t_max+1,
        n_runs,
        collective_variable=cv
    )
    if verbose: print("Finished SSA simulation.\n"
                      "Starting SDE simulation.")

    t_sde, x_sde = sample_cle(
        params,
        initial_state=x_init_shares[0],
        max_time = t_max,
        num_time_steps=t_max*simulation_resolution_sde*save_resolution,
        num_samples=n_runs,
        saving_offset=simulation_resolution_sde
    )
    if verbose: print("Finished SDE simulation.")

    assert len(t_ref) == len(t_sde)

    ws_distances = np.zeros((len(t_ref), n_states))

    for m in range(n_states):
        for i in range(len(t_sde)):

            ws_distances[i,m] = sp.stats.wasserstein_distance(x_ref[0,:,i,m], x_sde[:,i,m])

    return ws_distances



def simpletest(test=False):


    R = np.array([[0, .8, .2],
                  [.2, 0, .9],
                  [.8, .3, 0]])

    Rt = 0.01 * np.array([[0, .9, .8],
                          [.7, 0, .9],
                          [.9, .7, 0]])
    n_states = R.shape[0]

    if test:
        t_max = 10
        n_runs = 10
        n_nodes_list = [10]
    else:
        t_max = 100
        n_runs = 1000000
        n_nodes_list = [10, 100, 1000, 10000, 100000, 1000000]

    errs = []

    for n_nodes in n_nodes_list:
        x_init_shares, x_init_network = (
            create_equal_network_init_and_shares([.2, .5, .3], n_nodes))

        params = CNVMParameters(num_opinions=n_states,
                                num_agents=n_nodes,
                                r=R,
                                r_tilde=Rt,
                                alpha=1)
        diffs = compare_wasserstein(params, x_init_network, n_runs, t_max, verbose=True)
        errs.append(diffs)

        if test:
            plt.plot(diffs)
            plt.show()


    if not test:
        save_path = os.path.join(data_path, "ws_distance")
        np.savez(os.path.join(save_path, f"wasserstein_errs_{n_runs}nr.npy"),
                 errs=np.array(errs),
                 n_nodes_list=n_nodes_list)

    return





def main():


    args = parser.parse_args()
    test: bool = args.test

    if test:
        simpletest(test=True)

    return



if __name__=="__main__":
    main()