import datetime
import h5py

import scipy as sp
from utils.parameter_utils import *
from utils.computation.computation_utils import *
from utils.network_utils import *
import json
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


def run_wasserstein_test(
        n_states: int,
        rate_type: str,
        n_runs_mjp: int,
        n_runs_sde: int,
        batchsize_mjp: int,
        batchsize_sde: int,
        t_max: int,
        network_save_path: str,
        save_path: str = "",
        save_resolution=2,
        simulation_resolution_sde=20,
        process_id: str = "",
        delete_batches: bool = True
):

    # Network Initialization
    network, network_params = read_network(network_save_path)

    # Parameter Initialization
    parameter_generator = get_parameter_generator(rate_type, n_states)
    params, initial_rel_shares, name_rate_type = parameter_generator(network)

    # Result dir preparation
    run_name = f"ws_dist_{name_rate_type}_{network_params['network_name'].decode()}"
    path_save_dir = os.path.join(save_path, run_name)
    os.mkdir(path_save_dir)

    # Creating initial states
    initial_rel_shares, network_init = (
        create_equal_network_init_and_shares(initial_rel_shares, network_params["n_nodes"])
    )

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
        save_path_batch=path_save_dir,
        batch_id=process_id
    )

    # Computation of Wasserstein distance
    t, wasserstein_distances = compute_wasserstein_distance_from_batches(paths_batches_mjp, paths_batches_sde)

    if delete_batches:
        for file in paths_batches_mjp + paths_batches_sde:
            os.remove(file)

    # Saving Results
    with h5py.File(os.path.join(path_save_dir, run_name + ".hdf5"), "w") as file:
        ws_dist_group = file.create_group("wasserstein_distance")

        ws_dist_group.attrs["rate_type"] = rate_type
        ws_dist_group.attrs["n_states"] = n_states
        for attribute in network_params.keys():
            ws_dist_group.attrs[attribute] = network_params[attribute]
        ws_dist_group.attrs["params_str"] = str(params)
        ws_dist_group.attrs["n_runs_sde"] = n_runs_sde
        ws_dist_group.attrs["n_runs_mjp"] = n_runs_mjp
        ws_dist_group.attrs["sde_simulation_resolution"] = simulation_resolution_sde
        ws_dist_group.attrs["date_of_computation"] = datetime.datetime.now().isoformat()

        ws_dist_group.create_dataset("wasserstein_distance", data=wasserstein_distances)
        ws_dist_group.create_dataset("t", data=t)

    return


def standard_wasserstein_test(
        n_states: int,
        rate_type: str,
        network_save_path: str,
        process_id: str

):

    run_wasserstein_test(
        network_save_path=network_save_path,
        n_states=n_states,
        rate_type=rate_type,
        n_runs_sde=1000000,
        n_runs_mjp=1000000,
        batchsize_mjp=10000,
        batchsize_sde=100000,
        t_max=100,
        save_path=save_path_results,
        save_resolution=2,
        simulation_resolution_sde=20,
        process_id=process_id
    )

    return


def development_wasserstein_test():


    run_wasserstein_test(
        network_save_path="/home/manuel/Documents/code/data/test_data/ER_n100_p-crit-100.hdf5",
        n_states=3,
        rate_type="asymm",
        n_runs_sde=1000,
        n_runs_mjp=1000,
        batchsize_mjp=500,
        batchsize_sde=500,
        t_max=100,
        save_path=save_path_results,
        save_resolution=2,
        simulation_resolution_sde=20
    )
    return


def main():

    #  Argument Parsing ##########
    args = parser.parse_args()
    test: bool = args.test

    if not test:
        standard_wasserstein_test(
            args.n_states,
            args.rate_type,
            args.n_nodes,
            args.edge_probability,
            args.network_path,
            args.id
        )
    else:
        development_wasserstein_test()

    return


if __name__ == "__main__":
    main()
