import datetime
from typing import List
import scipy as sp
from utils.parameter_utils import WassersteinParameters, standard_ws_from_network_and_rate_type
from utils.computation_utils import compute_mjp_sde_runs
import h5py
import json
import numpy as np

import os
import argparse


# Load file containing all relevant paths for file saving and loading
with open("paths.json") as file:
    path_data = json.load(file)

data_path = path_data.get("data_path", "")
path_tmp_save = path_data.get("path_tmp_save", "")
save_path_results = path_data.get("save_path_results", "")

# Command Line Arguments ###############################################################################################
parser = argparse.ArgumentParser(
    description="This script tests the convergence rate of the diffusion approximation to a CNVM "
                "with respect to the Wasserstein distance."
)
mjp_parameters = parser.add_argument_group("Markov Jump Process parameters")
mjp_parameters.add_argument(
    "rate_type",
    type=str,
    help="Name of the parameter set that should be used. "
         "See `parameter_utils.py` for more exact specifications."
)
mjp_parameters.add_argument(
    "n_states",
    type=int,
    help="Number of states in the markov jump process on network. "
         "Corresponding rate_type parameter set needs to be specified in `parameter_utils.py`."
)
network_parameters = parser.add_argument_group("Network parameters")
network_parameters.add_argument(
    "--network_path",
    type=str,
    default=None,
    help="Path to a network that should be used for computation."
)
network_parameters.add_argument(
    "--n_nodes",
    type=int,
    default=None,
    help="Number of nodes in the network."
)
network_parameters.add_argument(
    "--edge_probability",
    type=float,
    default=None,
    help="Probability p for which an edge exists in the G(n,p) model. "
         "Set to 1 to use fully connected network."
         "Defaults to None"
)
computation_parameters = parser.add_argument_group("Computation Parameters")
computation_parameters.add_argument(
    "--id",
    type=str,
    default="",
    help="Identifying string that differentiates batches from concurrently running processes."
         "If several processes run at the same time use this string to differentiate them."
)
computation_parameters.add_argument(
    "--test",
    action="store_true",
    help="Set to test the script"
)






def load_batches(paths_batches: List[str]) -> tuple[np.ndarray, np.ndarray]:

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




def run_full_wasserstein_test(
        ws_params: WassersteinParameters,
        save_path: str = "",
        process_id: str = "",
        delete_batches: bool = True,
):

    save_dir_path = os.path.join(save_path, ws_params.run_name)
    if not os.path.isdir(save_dir_path):
        os.mkdir(save_dir_path)


    # Trajectory computation
    paths_batches_mjp, paths_batches_sde = compute_mjp_sde_runs(
        comp_params=ws_params,
        batch_save_path=save_dir_path,
        batch_id=process_id
    )

    # Computation of Wasserstein distance
    t, wasserstein_distances = compute_wasserstein_distance_from_batches(paths_batches_mjp, paths_batches_sde)

    if delete_batches:
        for entry in paths_batches_mjp + paths_batches_sde:
            os.remove(entry)

    # Saving Results
    save_wasserstein_result(
        os.path.join(save_dir_path, ws_params.run_name+".hdf5"),
        ws_params,
        wasserstein_distances,
        t
    )

    return


def save_wasserstein_result(
        path: str | os.PathLike,
        ws_params: WassersteinParameters,
        error_trajectory: np.ndarray,
        time_trajectory: np.ndarray
):
    if not path.endswith(".hdf5"):
        path += ".hdf5"


    with h5py.File(path, "w") as save_file:
        ws_dist_group = save_file.create_group("wasserstein_distance")

        ws_dist_group.attrs["rate_type"] = ws_params.rate_type
        ws_dist_group.attrs["n_states"] = ws_params.n_states
        for attribute in ws_params.network_params.keys():
            ws_dist_group.attrs[attribute] = ws_params.network_params[attribute]
        ws_dist_group.attrs["params_str"] = str(ws_params.cnvm_params)
        ws_dist_group.attrs["n_runs_sde"] = ws_params.n_runs_sde
        ws_dist_group.attrs["n_runs_mjp"] = ws_params.n_runs_mjp
        ws_dist_group.attrs["sde_simulation_resolution"] = ws_params.simulation_resolution_sde
        ws_dist_group.attrs["date_of_computation"] = datetime.datetime.now().isoformat()

        ws_dist_group.create_dataset("wasserstein_distance", data=error_trajectory)
        ws_dist_group.create_dataset("t", data=time_trajectory)

    return


def standard_wasserstein_test(
        n_states: int,
        rate_type: str,
        network_save_path: str,
        process_id: str

):

    save_path = save_path_results

    ws_params = standard_ws_from_network_and_rate_type(
        n_states=n_states,
        rate_type=rate_type,
        network_save_path=network_save_path,
    )

    run_full_wasserstein_test(ws_params, save_path=save_path, delete_batches=True)

    return




def main():

    #  Argument Parsing ##########
    args = parser.parse_args()
    test: bool = args.test

    if not test:
        standard_wasserstein_test(
            args.n_states,
            args.rate_type,
            args.network_path,
            "tmp"
        )
    else:
       pass
    return


if __name__ == "__main__":
    main()
