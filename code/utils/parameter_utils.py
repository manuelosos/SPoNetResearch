from sponet import CNVMParameters
from sponet.network_generator import NetworkGenerator
import networkx as nx
import numpy as np
import os

def parameter_set(
		transition_rate_matrix_monadic: np.ndarray,
		transition_rate_matrix_dyadic: np.ndarray,
		network_gen: NetworkGenerator,
		path_network: None | str
):
	"""
	Generic function for initializing parameter sets.
	:param transition_rate_matrix_monadic:
	:param transition_rate_matrix_dyadic:
	:param network_gen:
	:param path_network:
	:return:
	"""

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


def cnvm_3s_asymm(
		network_gen: NetworkGenerator,
		path_network: str | None = None
):

	R = np.array([[0, 0.8, 0.2],
	             [0.2, 0, 0.8],
	             [0.8, 0.2, 0]])
	Rt = np.array([[0, 0.01, 0.01],
	              [0.01, 0, 0.01],
	              [0.01, 0.01, 0]])

	x_init_shares = np.array([0.2, 0.5, 0.3])

	params, network_name = parameter_set(
		transition_rate_matrix_monadic=Rt,
		transition_rate_matrix_dyadic=R,
		network_gen=network_gen,
		path_network=path_network
	)

	return params, x_init_shares, "CNVM_3s_asymm"
