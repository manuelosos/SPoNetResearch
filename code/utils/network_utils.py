import numpy as np
from sponet.collective_variables import OpinionShares

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
