import datetime

import numpy as np
from sponet.network_generator import ErdosRenyiGenerator
import scipy as sp
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

# Command Line Arguments ###############################################################################################
parser = argparse.ArgumentParser(
    description="This script tests the convergence rate of the diffusion approximation to a CNVM "
                "with respect to the Wasserstein distance."
)
parser.add_argument("--test", action="store_true", help="Set to test the script")



def load_batches(paths_batches: List[str]) -> Tuple[np.ndarray, np.ndarray]:

    res = np.load(paths_batches[0])["x"]
    t = np.load(paths_batches[0])["t"]

    for path_batch in paths_batches[1:]:
        res = np.concatenate((res, np.load(path_batch)["x"]), axis=0)
    return t, res


def compute_wasserstein_distance_from_batches(
        paths_batches_mjp: List[str],
        paths_batches_sde: List[str],
        verbose=False
):

    t, trajectories_mjp = load_batches(paths_batches_mjp)
    t, trajectories_sde = load_batches(paths_batches_sde)

    n_time_steps = trajectories_mjp.shape[1]
    n_states = trajectories_mjp.shape[2]

    distances_wasserstein = np.empty((n_time_steps, n_states))

    if verbose:
        print("Starting Wasserstein distance computation...")

    for time_step in range(n_time_steps):
        for state in range(n_states):

            distances_wasserstein[time_step, state] = (
                sp.stats.wasserstein_distance(
                    trajectories_mjp[:, time_step, state],
                    trajectories_sde[:, time_step, state]
                )
            )
    if verbose:
        print("Wasserstein distance computation complete.")

    return t, distances_wasserstein


def run_wasserstein_test(
        n_nodes: int,
        edge_density: float,
        n_states: int,
        rate_type: str,
        n_runs_mjp: int,
        n_runs_sde: int,
        batchsize_mjp: int,
        batchsize_sde: int,
        t_max: int,
        save_path: str = "",
        save_resolution=2,
        simulation_resolution_sde=20,
        network_save_path: str | None = None
):

    # Parameter Initialization
    network_params = {"n_nodes": n_nodes, "edge_density": edge_density, "network_save_path": network_save_path}
    parameter_generator = get_parameter_generator(rate_type, n_states)
    params, initial_rel_shares, name_network, name_rate_type = parameter_generator(network_params)
    # Result dir preparation
    run_name = f"ws_dist_{name_rate_type}_{n_nodes}n_{name_network}"
    path_save_dir = os.path.join(save_path, run_name)
    os.mkdir(path_save_dir)

    # Creating initial states
    initial_rel_shares, network_init = (
        create_equal_network_init_and_shares(initial_rel_shares, n_nodes))

    # Trajectory computation
    paths_batches_mjp, paths_batches_sde = compute_mjp_sde_runs(
        params=params,
        x_init_network=network_init,
        n_runs_sde=n_runs_sde,
        n_runs_mjp=n_runs_mjp,
        t_max=t_max,
        save_resolution=save_resolution,
        simulation_resolution_sde=simulation_resolution_sde,
        batchsize_sde=batchsize_sde,
        batchsize_mjp=batchsize_mjp,
        save_path_batch=path_tmp_save
    )

    # Computation of Wasserstein distance
    t, wasserstein_distances = compute_wasserstein_distance_from_batches(paths_batches_mjp, paths_batches_sde)

    # Saving Results
    np.savez_compressed(
        os.path.join(path_save_dir, run_name),
        t=t,
        ws_distance=wasserstein_distances
        )
    with open(os.path.join(path_save_dir, run_name + ".txt"), "w") as f:
        f.write(str(params))
        f.write("\n")
        f.write(f"Number of runs MJP: {n_runs_mjp}\n")
        f.write(f"Number of runs SDE: {n_runs_sde}\n")
        f.write(datetime.datetime.now().isoformat())

    return


def standard_wasserstein_test(
        n_nodes: int,
        edge_density: float,
        n_states: int,
        rate_type: str,

):

    run_wasserstein_test(
        n_nodes=n_nodes,
        edge_density=edge_density,
        n_states=n_states,
        rate_type=rate_type,
        n_runs_sde=1000000,
        n_runs_mjp=1000000,
        batchsize_mjp=10000,
        batchsize_sde=100000,
        t_max=100,
        save_path=save_path_results,
        save_resolution=2,
        simulation_resolution_sde=20
    )

    return


def main():


    args = parser.parse_args()
    test: bool = args.test

    if test:
        run_wasserstein_test(
            n_nodes=100,
            edge_density=0.9,
            n_states=3,
            rate_type="asymm",
            n_runs_mjp=100,
            n_runs_sde=100,
            batchsize_mjp=10,
            batchsize_sde=10,
            t_max=10
        )
    return



if __name__ == "__main__":
    main()
