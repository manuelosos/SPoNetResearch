import datetime

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
        save_path: str = "",
        save_resolution=2,
        simulation_resolution_sde=20,
        network_save_path: str | None = None,
        n_nodes: int | None = None,
        edge_density: float | None = None,
        process_id: str = ""
):

    # Parameter Initialization
    network_params = {"n_nodes": n_nodes, "edge_density": edge_density, "network_save_path": network_save_path}
    parameter_generator = get_parameter_generator(rate_type, n_states)
    params, initial_rel_shares, name_network, name_rate_type = parameter_generator(network_params)

    n_nodes = params.num_agents

    # Result dir preparation
    run_name = f"ws_dist_{name_rate_type}_{name_network}"
    path_save_dir = os.path.join(save_path, run_name)
    os.mkdir(path_save_dir)

    # Creating initial states
    initial_rel_shares, network_init = (
        create_equal_network_init_and_shares(initial_rel_shares, n_nodes)
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
        save_path_batch=path_tmp_save,
        batch_id=process_id

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
        n_states: int,
        rate_type: str,
        n_nodes: int | None,
        edge_probability: float | None,
        network_save_path: str | None,
        process_id: str

):

    run_wasserstein_test(
        n_nodes=n_nodes,
        edge_density=edge_probability,
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
        n_nodes=100,
        edge_density=0.8,
        network_save_path=None,
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
