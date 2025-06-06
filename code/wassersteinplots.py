import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
import scipy as sp
from utils.network_utils import get_available_networks

with open("paths.json") as file:
    data = json.load(file)
data_path = data.get("data_path")


def plot_wasserstein_distance_vs_network_size(
        ref_state=0
):

    data = np.load(os.path.join(data_path, "ws_distance/wasserstein_errs_100000s.npz"))

    node_list = data.get("n_nodes_list")
    errs = data.get("errs")
    print(errs.shape)

    plot_err = np.empty(len(node_list))
    for i, n_nodes in enumerate(node_list):
        plot_err[i] = np.max(errs[i,:,ref_state])

    error_bound = np.log(node_list)/node_list

    fig, ax = plt.subplots()

    ax.plot( node_list, errs[:, 10, ref_state], marker="o", label="wasserstein distance")
    ax.plot(node_list, error_bound)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.show()

    return


def test_network_creation():

    res = sp.io.mmread("ER_p-crit1000_N1000")
    print(res.toarray().shape)


def main():
    return


if __name__ == "__main__":
    # plot_wasserstein_distance_vs_network_size()
    print(get_available_networks("utils/"))
