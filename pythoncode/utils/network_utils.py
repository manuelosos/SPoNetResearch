from typing import Tuple, Dict
from dataclasses import dataclass
import re
import numpy as np
from sponet.collective_variables import OpinionShares
from numba import njit, prange
import networkx as nx
import h5py
import os
from scipy import sparse as sp_sparse



def get_available_networks(
        path: str,
        file_ending: str = ".hdf5"
) -> list[str]:
    """
    Returns a list of all available networks in a directory with absolute paths.
    :param path:
    Whether to return only the names of the networks and not the absolute paths.
    :param file_ending:
    Ending string should start with `.`.
    :return:
    """
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a directory")

    dir_list = os.listdir(path)
    return_list = []

    for entry in dir_list:
        if os.path.isdir(os.path.join(path, entry)):
            continue
        if entry.endswith(file_ending):
            return_list.append(os.path.join(path, entry))

    return return_list


def get_available_network_params(path: str):
    network_paths = get_available_networks(path)

    parameter_list = []
    for network_path in network_paths:
        try:
            parameter_list.append(read_network(network_path, return_only_parameters=True))
        except OSError:
            print(f"Could not read {network_path}")
            continue

    return parameter_list


def read_network(
        save_path: str,
        return_only_parameters: bool = False
):
    with h5py.File(save_path, "r") as file:

        parameters = file["network_data"].attrs
        parameters = dict(zip(parameters.keys(), parameters.values()))

        if return_only_parameters:
            return parameters

        if "adjacency_matrix" in file["network_data"].keys():
            adjacency_matrix = file["network_data"]["adjacency_matrix"]
            return nx.from_numpy_array(np.array(adjacency_matrix)), parameters

        elif "adjacency_matrix_col_ptr" in list(file["network_data"].keys()) and "adjacency_matrix_row_ptr" in list(file["network_data"].keys()):

            col_ptr = np.array(file["network_data"]["adjacency_matrix_col_ptr"])
            row_ptr = np.array(file["network_data"]["adjacency_matrix_row_ptr"])

            data = np.ones(len(col_ptr), dtype=np.uint8)
            adj_mat = sp_sparse.csr_matrix((data, col_ptr, row_ptr))
            return nx.from_scipy_sparse_array(adj_mat), parameters

    raise ValueError("cannot read network")



def save_network(
        save_path,
        network: nx.Graph,
        meta_data: dict[str, np.ndarray] = None
):
    adj_matrix = nx.to_numpy_array(network)

    np.savez_compressed(save_path, adj_matrix=adj_matrix, **meta_data)
    return


def get_network_params_from_name(
    network_name: str
) -> tuple[int, float]:
    """
    Extracts number of nodes and edge probability from the network name.
    :param network_name:
    :return:
    """

    segments = network_name.split("_")


    # Network model
    if segments[0] == "CN":
        network_model = "CN"
    elif segments[0] == "ER":
        network_model = "ER"
    else:
        raise ValueError(f"{network_name} is not a valid network name")

    # edge probability
    match = re.match(r"^(.*?)(\d+)$", segments[1])

    if match:
        prefix = match.group(1)
        number = match.group(2)

        if prefix == "p":
            edge_probability = number/100
        elif prefix == "p-crit":
            edge_probability = np.log(int(number))/int(number)
        else:
            raise ValueError(f"{network_name} is not a valid network name")
    else:
        raise ValueError(f"{network_name} is not a valid network name")

    # n_nodes

    match = re.match(r"^(.*?)(\d+)$", segments[2])
    if match:
        prefix = match.group(1)
        number = match.group(2)

        if prefix == "N":
            n_nodes = int(number)
        else:
            raise ValueError(f"{network_name} is not a valid network name")
    else:
        raise ValueError(f"{network_name} is not a valid network name")


    return n_nodes, edge_probability


def create_network_init(shares, n_nodes):

    total_share_counts = np.array([int(share*n_nodes) for share in shares])

    count_diff = n_nodes - sum(total_share_counts)
    if count_diff != 0:

        for i in range(np.abs(count_diff)):
            total_share_counts[i % n_nodes] += np.sign(count_diff)

    assert sum(total_share_counts) == n_nodes

    x_init = np.concatenate([i * np.ones(total_share_counts[i]) for i in range(len(shares))])

    return x_init


def create_equal_network_init_and_shares(
        shares: np.ndarray,
        n_nodes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a network state with shares as close as possible to the proposed shares.
    The shares are then updated accordingly to ensure equal initial conditions if one
    tests CLE against network process.
    :rtype: Tuple[np.ndarray, np.ndarray]
    :param shares:
        Shares of the state
    :param n_nodes:
        Number of nodes of the network
    """

    n_states = len(shares)
    cv = OpinionShares(n_states, normalize=True)
    network_state = create_network_init(shares, n_nodes)
    shares = cv(np.array([network_state]))

    return shares, network_state


@njit(parallel=True, cache=True)
def compute_propensity_difference_trajectory(neighbor_list, x_traj, rel_shares, R):

    res = np.zeros(x_traj.shape[0])
    for i in prange(x_traj.shape[0]):
        res[i] = compute_propensity_difference(neighbor_list, x_traj[i], rel_shares[i], R)

    return res



@njit(cache=True)
def compute_propensity_difference(neighbor_list, x, rel_shares, R):
    """
    Computes the difference Delta^G_ell for ER networks as described in Lueckes Thesis.
    """
    n_states = R.shape[0]
    state_propensity_differences = np.zeros((n_states, n_states))
    n_nodes = len(neighbor_list)
    rel_shares_products = np.outer(rel_shares, rel_shares)

    for k in range(n_nodes):

        degree = len(neighbor_list[k])
        origin_state = x[k]
        relshares_neighborhood = np.zeros(n_states)

        for neighbor in neighbor_list[k]:
            relshares_neighborhood[x[neighbor]] += 1
        relshares_neighborhood /= degree


        for target_state in range(n_states):
            if target_state == origin_state:
                continue

            state_propensity_differences[origin_state, target_state] += relshares_neighborhood[target_state]

    state_propensity_differences = np.abs(state_propensity_differences/n_nodes - rel_shares_products) * R

    return np.max(state_propensity_differences)


if __name__ == "__main__":
    print(res :=read_network("/home/manuel/Documents/code/data/test_data/ER_n5_p-crit-5.hdf5"))
