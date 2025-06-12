from typing import Callable

import sponet
from sponet import CNVMParameters
from sponet.network_generator import NetworkGenerator
import networkx as nx
import numpy as np
from .network_utils import read_network
import os
from dataclasses import dataclass


@dataclass()
class NetworkParameters:
	name: str
	network_model: str
	n_nodes: int


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


def get_parameter_generator(rate_type: str, n_states: int) -> Callable:

	rate_key = f"{n_states}s_{rate_type}"

	function_dict = {
		"3s_asymm": cnvm_3s_asymm,
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