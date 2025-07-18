from typing import Callable
from sponet import CNVMParameters
import networkx as nx
import numpy as np
from .network_utils import read_network, create_equal_network_init_and_shares
from dataclasses import dataclass


@dataclass
class WassersteinParameters:
	n_states: int
	rate_type: str
	network: nx.Graph
	network_params: dict
	t_max: int
	n_runs_mjp: int
	n_runs_sde: int
	batchsize_mjp: int
	batchsize_sde: int
	save_resolution: int
	simulation_resolution_sde: int

	def __post_init__(self):
		if self.n_runs_mjp % self.batchsize_mjp != 0:
			raise ValueError("Number of mjp runs must be divisible by the batchsize")
		if self.n_runs_sde % self.batchsize_sde != 0:
			raise ValueError("Number of sde runs must be divisible by the batchsize")

		# CNVM parameter Initialization
		parameter_generator = get_parameter_generator(self.rate_type, self.n_states)
		self.cnvm_params, initial_rel_shares, name_rate_type = parameter_generator(self.network)

		self.initial_rel_shares, self.network_init = (
			create_equal_network_init_and_shares(initial_rel_shares, self.network_params["n_nodes"])
		)
		self.run_name: str = f"ws_dist_CNVM_{self.n_states}s_{self.rate_type}_{self.network_params['network_name'].decode()}"


def standard_ws_from_network_and_rate_type(
		n_states: int,
		rate_type: str,
		network_save_path: str,
) -> WassersteinParameters:

	# Network Initialization
	network, network_params = read_network(network_save_path)

	return WassersteinParameters(
		n_states=n_states,
		rate_type=rate_type,
		network=network,
		network_params=network_params,
		t_max=200,
		n_runs_mjp=1_000_000,
		n_runs_sde=1_000_000,
		batchsize_mjp=10_000,
		batchsize_sde=100_000,
		save_resolution=2,
		simulation_resolution_sde=20
	)


def standard_ws_with_new_network_from_rate_type(
		n_nodes: int,
		edge_probability: float,
		n_states: int,
		rate_type: str,
		n_runs: int = 1_000_000,
		t_max: int = 200,
		batchsize_mjp: int = 10_000,
		batchsize_sde: int = 100_000,
) -> WassersteinParameters:


	network = nx.erdos_renyi_graph(n_nodes, edge_probability)
	network_parameters = {
		"edge_probability": edge_probability,
		"edge_probability_name": "no given name",
		"network_name": f"ER_n{n_nodes}".encode(),
		"n_nodes": n_nodes,
		"network_model": "erdos_renyi",
	}

	return WassersteinParameters(
		n_states=n_states,
		rate_type=rate_type,
		network=network,
		network_params=network_parameters,
		t_max=t_max,
		n_runs_mjp=n_runs,
		n_runs_sde=n_runs,
		batchsize_mjp=batchsize_mjp,
		batchsize_sde=batchsize_sde,
		save_resolution=2,
		simulation_resolution_sde=20
	)


def _get_parameter_set(
		transition_rate_matrix_monadic: np.ndarray,
		transition_rate_matrix_dyadic: np.ndarray,
		network: nx.Graph
) -> CNVMParameters:
	"""
	Generic function for initializing parameter sets.
	:param transition_rate_matrix_monadic:
	:param transition_rate_matrix_dyadic:
	:param network:
	:return: CNVMParameters
	"""

	mjp_params = CNVMParameters(
		num_opinions=transition_rate_matrix_monadic.shape[0],
		network=network,
		r=transition_rate_matrix_dyadic,
		r_tilde=transition_rate_matrix_monadic,
		alpha=1
	)
	return mjp_params


def test_ws_from_network_and_rate_type(
	n_states: int,
	rate_type: str,
	network_save_path: str,
	verbose = True
	) -> WassersteinParameters:

	# Network Initialization
	network, network_params = read_network(network_save_path)

	return WassersteinParameters(
		n_states=n_states,
		rate_type=rate_type,
		network=network,
		network_params=network_params,
		t_max=2,
		n_runs_mjp=40,
		n_runs_sde=100,
		batchsize_mjp=10,
		batchsize_sde=50,
		save_resolution=2,
		simulation_resolution_sde=20
	)


def get_parameter_generator(rate_type: str, n_states: int) -> Callable:

	rate_key = f"{n_states}s_{rate_type}"

	function_dict = {
		"2s_symm": cnvm_2s_symmetric,
		"3s_asymm": cnvm_3s_asymm,
		"3s_symm": cnvm_3s_symmetric,
		"3s_almost-symm": cnvm_3s_almost_symm,


	}

	return function_dict[rate_key]


def cnvm_3s_asymm(
		network: nx.Graph,
):

	R = np.array([[0, 0.8, 0.2],
	             [0.2, 0, 0.8],
	             [0.8, 0.2, 0]])
	Rt = np.array([[0, 0.01, 0.01],
	              [0.01, 0, 0.01],
	              [0.01, 0.01, 0]])

	x_init_shares = np.array([0.2, 0.5, 0.3])

	params = _get_parameter_set(
		transition_rate_matrix_monadic=Rt,
		transition_rate_matrix_dyadic=R,
		network=network
	)

	return params, x_init_shares, "CNVM_3s_asymm"


def cnvm_3s_almost_symm(network: nx.Graph):

	R = np.array([[0, 1, 0.99],
	              [0.99, 0, 1],
	              [1, 0.99, 0]])
	Rt = np.array([[0, 0.01, 0.01],
	               [0.01, 0, 0.01],
	               [0.01, 0.01, 0]])

	x_init_shares = np.array([0.2, 0.3, 0.5])

	params = _get_parameter_set(
		transition_rate_matrix_monadic=Rt,
		transition_rate_matrix_dyadic=R,
		network=network

	)
	return params, x_init_shares, "CNVM_3s_almost-symm"


def cnvm_3s_symmetric(network: nx.Graph):
	R = np.array([[0, 1, 1],
	              [1, 0, 1],
	              [1, 1, 0]])
	Rt = np.array([[0, 0.01, 0.01],
	               [0.01, 0, 0.01],
	               [0.01, 0.01, 0]])

	x_init_shares = np.array([1/3, 1/3, 1/3])

	params = _get_parameter_set(
		transition_rate_matrix_monadic=Rt,
		transition_rate_matrix_dyadic=R,
		network=network

	)
	return params, x_init_shares, "CNVM_3s_symm"


def cnvm_2s_symmetric(network: nx.Graph):
	R = np.array([[0, 1],
	              [1, 0]])
	Rt = np.array([[0, 0.01],
	               [0.01, 0]])

	x_init_shares = np.array([1/2, 1/2])

	params = _get_parameter_set(
		transition_rate_matrix_monadic=Rt,
		transition_rate_matrix_dyadic=R,
		network=network

	)
	return params, x_init_shares, "CNVM_2s_symm"
