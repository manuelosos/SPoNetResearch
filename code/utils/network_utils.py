import numpy as np
from sponet.collective_variables import OpinionShares
from numba import njit, prange

def create_network_init(shares, n_nodes):

    total_share_counts = np.array([int(share*n_nodes) for share in shares])

    count_diff = n_nodes - sum(total_share_counts)
    if count_diff != 0:

        for i in range(np.abs(count_diff)):
            total_share_counts[i % n_nodes] += np.sign(count_diff)

    assert sum(total_share_counts) == n_nodes

    x_init = np.concatenate([i * np.ones(total_share_counts[i]) for i in range(len(shares))])

    return x_init


def create_equal_network_init_and_shares(shares, n_nodes):
    """
    Creates a network state with shares as close as possible to the proposed shares.
    The shares are then updated accordingly to ensure equal initial condiitions if one
    tests CLE against network process.
    :param shares:
        Shares of the state
    :param n_nodes:
        Number of nodes of the network
    :return:
        Tuple[np.array, np.array]
    """


    n_states = len(shares)
    cv = OpinionShares(n_states)
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