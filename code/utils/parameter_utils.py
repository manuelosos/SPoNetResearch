from typing import Callable

import sponet
from sponet import CNVMParameters
from sponet.network_generator import NetworkGenerator
import networkx as nx
import numpy as np
import os


def _get_parameter_set(
		transition_rate_matrix_monadic: np.ndarray,
		transition_rate_matrix_dyadic: np.ndarray,
		network_params: dict,
):
	"""
	Generic function for initializing parameter sets.
	:param transition_rate_matrix_monadic:
	:param transition_rate_matrix_dyadic:
	:param network_params:
	:return:
	"""

	# unpacking network parameters
	n_nodes = network_params["n_nodes"]
	edge_density_erdos_renyi = network_params.get("edge_density", 1)
	path_network = network_params.get("network_save_path", None)

	if edge_density_erdos_renyi == 1:

		params = CNVMParameters(
			num_opinions=transition_rate_matrix_monadic.shape[0],
			num_agents=n_nodes,
			r=transition_rate_matrix_dyadic,
			r_tilde=transition_rate_matrix_monadic,
			alpha=1
		)

		network_name = f"CN_N{n_nodes}"
		return params, network_name

	if edge_density_erdos_renyi < 1:
		network_gen = sponet.network_generator.ErdosRenyiGenerator(n_nodes, edge_density_erdos_renyi)
	else:
		raise NotImplementedError

	if path_network is not None:
		adjacency_matrix = np.load(path_network)
		network = nx.from_numpy_array(adjacency_matrix)

		params = CNVMParameters(
			num_opinions=transition_rate_matrix_monadic.shape[0],
			network=network,
			r=transition_rate_matrix_dyadic,
			r_tilde=transition_rate_matrix_monadic,
			alpha=1
		)
	else:
		params = CNVMParameters(
			num_opinions=transition_rate_matrix_monadic.shape[0],
			network_generator=network_gen,
			r=transition_rate_matrix_dyadic,
			r_tilde=transition_rate_matrix_monadic,
			alpha=1
		)

	network_name = network_gen.abrv()

	return params, network_name


def get_parameter_generator(rate_type: str, n_states: int) -> Callable:

	rate_key = f"{n_states}s_{rate_type}"

	function_dict = {
		"3s_asymm": cnvm_3s_asymm,
	}

	return function_dict[rate_key]


def cnvm_3s_asymm(
		network_params: dict,
):

	R = np.array([[0, 0.8, 0.2],
	             [0.2, 0, 0.8],
	             [0.8, 0.2, 0]])
	Rt = np.array([[0, 0.01, 0.01],
	              [0.01, 0, 0.01],
	              [0.01, 0.01, 0]])

	x_init_shares = np.array([0.2, 0.5, 0.3])

	params, network_name = _get_parameter_set(
		transition_rate_matrix_monadic=Rt,
		transition_rate_matrix_dyadic=R,
		network_params=network_params,
	)

	return params, x_init_shares, network_name, "CNVM_3s_asymm"
