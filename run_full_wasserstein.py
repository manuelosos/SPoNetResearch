import json
import argparse
import os

from pythoncode.utils import parameter_utils
from pythoncode.utils.parameter_utils import WassersteinParameters, standard_ws_from_network_and_rate_type
from pythoncode.utils.computation_utils import compute_mjp_sde_runs
from pythoncode.wasserstein.wasserstein import compute_wasserstein_distance_from_batches, save_wasserstein_result

# Load file containing all relevant paths for file saving and loading
with open("paths.json") as file:
	path_data = json.load(file)

data_path = path_data.get("data_path", "")
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


def standard_wasserstein_test(
		n_states: int,
		rate_type: str,
		network_save_path: str,
		process_id: str

):

	save_path = os.path.join(save_path_results, "ws_distance")

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
