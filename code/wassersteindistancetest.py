import sponet
import logging
import numpy as np
from logging import log
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

# Load file containing all relevant paths for file saving and loading
with open("paths.json") as file:
    data = json.load(file)

data_path = data.get("data_path", "")
logging_path = data.get("logging_path", "")


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(os.path.join(logging_path, "wasserstein.log"))
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

parser = argparse.ArgumentParser(
    description="This script tests the convergence rate of the Diffusion approximation to a CNVM "
                "with respect to the Wasserstein distance."
)
parser.add_argument("--test", action="store_true", help="Set to test the script")


def run_wasserstein_test():


    return


def compare_wasserstein(
        params,
        x_init_network,
        n_runs,
        t_max,
        save_resolution = 2,
        simulation_resolution_sde = 20,
        verbose = False
):
    """
    Computes the trajectories of the Markov jump process on network and the corresponding diffusion approximation
    and computes the wasserstein distance.
    :param params:
    :param x_init_network:
    :param n_runs:
    :param t_max:
    :param save_resolution:
    :param simulation_resolution_sde:
    :param verbose:
    :return:
    """



    ws_distances = np.zeros((len(t_ref), n_states))

    for m in range(n_states):
        for i in range(len(t_sde)):
            ws_distances[i, m] = sp.stats.wasserstein_distance(x_ref[0, :, i, m], x_sde[:, i, m])

    return ws_distances



def convergencetest(test=False):


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
        t_max = 50
        n_runs = 100000
        n_nodes_list = [10, 100, 1000, 10000, 100000]

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



        save_path = os.path.join(data_path, "ws_distance")
        np.savez(os.path.join(save_path, f"wasserstein_errs_{n_runs}s"),
                errs=np.array(errs),
                n_nodes_list=n_nodes_list
                )
        print(f"{n_nodes} done")
        logger.info(f"Wasserstein run for {n_nodes} nodes and {n_runs} runs done.")
    return





def main():


    args = parser.parse_args()
    test: bool = args.test

    if test:
        convergencetest(test=True)
        return

    convergencetest(test=False)

    return



if __name__=="__main__":
    main()