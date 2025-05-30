import sponet
import datetime
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
from utils.parameter_utils import *
from utils.computation_utils import *
import json
import os
import argparse

# Load file containing all relevant paths for file saving and loading
with open("paths.json") as file:
    path_data = json.load(file)

data_path = path_data.get("data_path", "")
path_tmp_save = path_data.get("path_tmp_save", "")
save_path_results = path_data.get("save_path_results", "")
logging_path = path_data.get("logging_path", "")


# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# file_handler = logging.FileHandler(os.path.join(logging_path, "wasserstein.log"))
# formatter = logging.Formatter('%(asctime)s - %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

parser = argparse.ArgumentParser(
    description="This script tests the convergence rate of the Diffusion approximation to a CNVM "
                "with respect to the Wasserstein distance."
)
parser.add_argument("--test", action="store_true", help="Set to test the script")



def load_batches(paths_batches: List[str]) -> np.ndarray:

    res = np.load(paths_batches[0])["x"]

    for path_batch in paths_batches[1:]:
        res = np.concatenate((res, np.load(path_batch)["x"]), axis=0)
    return res


def run_wasserstein_test(
        n_nodes: int,
        n_runs_sde: int,
        n_runs_mjp: int,
        batchsize_mjp: int,
        batchsize_sde: int,
        t_max: int,
        save_path: str = ""
):


    network_gen = sponet.network_generator.ErdosRenyiGenerator(n_nodes, 0.9)
    params, initial_rel_shares, name_rate_type = cnvm_3s_asymm(network_gen)

    initial_rel_shares, network_init = (
        create_equal_network_init_and_shares(initial_rel_shares, n_nodes))


    paths_batches_mjp, paths_batches_sde = compute_mjp_sde_runs(
        params=params,
        x_init_network=network_init,
        n_runs_sde=n_runs_sde,
        n_runs_mjp=n_runs_mjp,
        t_max=t_max,
        save_resolution=2,
        simulation_resolution_sde=20,
        batchsize_sde = batchsize_sde,
        batchsize_mjp = batchsize_mjp,
        save_path_batch=path_tmp_save
    )


    wasserstein_distances = compute_wasserstein_distance_from_batches(paths_batches_mjp, paths_batches_sde)

    run_name = f"ws_dist_{name_rate_type}_{n_nodes}n_{network_gen.abrv()}"
    path_save_dir = os.path.join(save_path, run_name)

    os.mkdir(path_save_dir)
    np.save(os.path.join(path_save_dir, run_name), wasserstein_distances)
    with open(os.path.join(path_save_dir, run_name + ".txt"), "w") as f:
        f.write(str(params))
        f.write("\n")
        f.write(datetime.datetime.now().isoformat())

    return


def compute_wasserstein_distance_from_batches(
        paths_batches_mjp: List[str],
        paths_batches_sde: List[str],
        verbose=False
):

    trajectories_mjp = load_batches(paths_batches_mjp)
    trajectories_sde = load_batches(paths_batches_sde)

    n_time_steps = trajectories_mjp.shape[1]
    n_states = trajectories_mjp.shape[2]

    distances_wasserstein = np.empty((n_time_steps, n_states))

    for time_step in range(n_time_steps):
        for state in range(n_states):

            distances_wasserstein[time_step, state] = (
                sp.stats.wasserstein_distance(
                    trajectories_mjp[:, time_step, state],
                    trajectories_sde[:, time_step, state]
                )
            )

    return distances_wasserstein


def main():


    args = parser.parse_args()
    test: bool = args.test

    if test:
        run_wasserstein_test(
            n_nodes=100,
            n_runs_mjp=100,
            n_runs_sde=100,
            batchsize_mjp=1000,
            batchsize_sde=1000,
            t_max=10
        )
    return



if __name__=="__main__":
    main()