import time
import argparse
import os
import numpy as np
import networkx as nx
from sponet.network_generator import *
from utils.network_utils import save_network
import json
import multiprocessing as mp
import itertools


# Initializing Paths
with open("paths.json") as file:
	path_data = json.load(file)

data_path = path_data.get("data_path", "")
save_path_results = path_data.get("save_path_networks", "")

# Initializing command line arguments
parser = argparse.ArgumentParser(
	description="This script creates a network of given size and edge density "
	            "and saves it in the save_path_results folder"
)

parser.add_argument(
	"n_nodes",
	type=int,
	help="The size of the network"
)
parser.add_argument(
	"edge_density",
	type=float,
	help="Edge density of the network."
)
parser.add_argument(
	"-fni", "--force_no_isolates",
	action="store_true",
	help="Set to force no isolates in the creation process. "
	     "May increase generation time substantially."
)
parser.add_argument(
	"-e", "--ensemble",
	action="store_true",
	help="Creates several networks using parameters specified in the code. "
	     "Networks are created at once using multiprocessing."
)
parser.add_argument(
	"--test",
	action="store_true",
	help="Set to test the ensemble mechanic with test values."
)


def generate_network(
		n_nodes: int,
		edge_density: float,
		force_no_isolates: bool,
		verbose = False
):

	if edge_density >= 1:
		raise ValueError("edge_density must be < 1")
	if n_nodes <= 0:
		raise ValueError("n_nodes must be > 0")
	if edge_density < np.log(n_nodes)/n_nodes:
		raise ValueError(f"Edge density must be larger than {np.log(n_nodes)/n_nodes}.")

	network_gen = ErdosRenyiGenerator(n_nodes, edge_density, force_no_isolates=force_no_isolates)

	start_time = time.time()

	network = network_gen()

	network_name = network_gen.abrv()
	save_network(os.path.join(save_path_results, network_name), network)

	end_time = time.time()
	elapsed_time = end_time - start_time
	if verbose:
		print(f"Elapsed time: {elapsed_time}")

	return


if __name__ == '__main__':
	args = parser.parse_args()

	n_nodes = args.n_nodes
	edge_density = args.edge_density
	force_no_isolates = args.force_no_isolates
	ensemble = args.ensemble
	test = args.test

	if ensemble:

		list_n_nodes = np.array([100, 1000, 10000, 100000, 1000000])

		if test:
			list_n_nodes = np.array([10, 11, 12])

		list_edge_densities = np.log(list_n_nodes) / list_n_nodes

		arguments = []
		for n, p in itertools.product(list_n_nodes, list_edge_densities):

			if p < np.log(n)/n:
				continue
			else:
				arguments.append([n, p, True])


		with mp.Pool(min(len(arguments), mp.cpu_count())) as pool:
			pool.starmap(generate_network, arguments)

	else:
		generate_network(n_nodes, edge_density, force_no_isolates)





