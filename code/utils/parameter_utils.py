from typing import Callable

import sponet
from sponet import CNVMParameters
from sponet.network_generator import NetworkGenerator
import networkx as nx
import numpy as np
from .network_utils import read_network
import os


def _get_parameter_set(
		transition_rate_matrix_monadic: np.ndarray,
		transition_rate_matrix_dyadic: np.ndarray,
		network_params: dict,
) -> tuple[CNVMParameters, str]:
	"""
	Generic function for initializing parameter sets.
	:param transition_rate_matrix_monadic:
	:param transition_rate_matrix_dyadic:
	:param network_params:
		If path network is contained, then the network will be loaded from the path.
		The name of the network will be used as is.
	:return tuple[CNVMParameters, network_name]:
	"""

	# unpacking network parameters
	n_nodes = network_params["n_nodes"]
	edge_density_erdos_renyi = network_params["edge_density"]
	path_network = network_params["network_save_path"]

	if path_network is not None:  # Load existing network
		name_network = os.path.basename(path_network)
		network, pre_gen_network_params = read_network(path_network)

		edge_density_erdos_renyi = pre_gen_network_params.get("edge_probability", 1)

		params = CNVMParameters(
			num_opinions=transition_rate_matrix_monadic.shape[0],
			network=network,
			r=transition_rate_matrix_dyadic,
			r_tilde=transition_rate_matrix_monadic,
			alpha=1
		)
		return params, name_network


	assert n_nodes is not None
	if edge_density_erdos_renyi == 1:
		# If edge probability is 1 complete networks are used and no network gen needs to be used

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
		# Erdos Renyi Network

		network_gen = sponet.network_generator.ErdosRenyiGenerator(n_nodes, edge_density_erdos_renyi)
		network_name = network_gen.abrv()

		params = CNVMParameters(
			num_opinions=transition_rate_matrix_monadic.shape[0],
			network_generator=network_gen,
			r=transition_rate_matrix_dyadic,
			r_tilde=transition_rate_matrix_monadic,
			alpha=1
		)

		return params, network_name


	raise NotImplementedError


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
