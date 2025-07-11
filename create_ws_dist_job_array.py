import argparse
import os
import os.path as osp
from pythoncode.utils.network_utils import read_network
parser = argparse.ArgumentParser(
	description="Creates a .txt file containing parameters for a job array executing wassersteindistancetest.py for "
	            "given parameters on available networks."
)
parser.add_argument(
	"rate_type",
	type=str,
	help="Rate type of the Markov jump process. See parameter.utils for more details."
)
parser.add_argument(
	"n_states",
	type=int,
	help="Number of states in the Markov jump process."
)
parser.add_argument(
	"network_path",
	type=str,
	help="Path to a dir where networks are saved."
)
parser.add_argument(
	"result_save_path",
	type=str,
	help="Path to the directory where results of Wasserstein tests are saved."
)

parser.add_argument(
	"--save_path",
	type=str,
	help="Path to dir where the parameter file should be saved.",
	default="."
)


def get_available_networks(
		path: str | os.PathLike,
):
	network_path_list = []
	for file in os.listdir(path):
		if file.endswith(".hdf5"):
			network_path_list.append(osp.join(path, file))

	return network_path_list



def create_array_parameter_file(
	rate_type: str,
	n_states: int,
	network_path: str | os.PathLike,
	result_save_path: str | os.PathLike,
	save_path: str | os.PathLike,
):

	network_paths = get_available_networks(network_path)

	counter = 0
	with open(os.path.join(save_path, "ws_distance_parameters.txt"), "w") as f:
		for network_path in network_paths:
			parameters = read_network(network_path, return_only_parameters=True)
			if parameters["n_nodes"] >= 50_000:
				continue
			counter += 1
			f.write(f"{rate_type} {n_states} --network_path={network_path}\n --result_save_path={result_save_path}\n")

	print(f"created {len(network_paths)} jobs.")


	return


def main():
	args = parser.parse_args()

	create_array_parameter_file(
		args.rate_type,
		args.n_states,
		args.network_path,
		args.result_save_path,
		args.save_path

	)
	return


if __name__ == '__main__':
	main()
