import argparse
import numpy as np
import os
import os.path as osp

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

def get_available_networks(
		path: str | os.PathLike,
):
	network_path_list = []
	for file in os.listdir(path):
		if file.endswith(".hdf5"):
			network_path_list.append(osp.abspath(file))

	return network_path_list


def create_array_parameter_file(
	rate_type: str,
	n_states: int,
	network_path: str | os.PathLike
):

	network_paths = get_available_networks(network_path)

	with open("parameters/ws_distance_parameters.txt", "w") as f:
		for network_path in network_paths:
			f.write(f"\"{rate_type}\" {n_states} \"{network_path}\"\n")
	print(f"created {len(network_paths)} jobs.")


	return


def main():
	args = parser.parse_args()

	create_array_parameter_file(
		args.rate_type,
		args.n_states,
		args.network_path,
	)
	return


if __name__ == '__main__':
	main()