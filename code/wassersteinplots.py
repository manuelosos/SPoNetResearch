import json
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
import scipy as sp
from utils.network_utils import get_available_networks, read_network

with open("paths.json") as file:
    data = json.load(file)
data_path = data.get("data_path")


def plot_wasserstein_distance_vs_network_size(
        ref_state=0
):

    data = np.load(os.path.join(data_path, "ws_distance/wasserstein_errs_100000s.npz"))

    node_list = data.get("n_nodes_list")
    errs = data.get("errs")

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


def read_run():
    
    with h5py.File("/home/manuel/Documents/code/SpoNetResearch/code/ws_dist_CNVM_3s_asymm_ER_n100_p-crit-100/ws_dist_CNVM_3s_asymm_ER_n100_p-crit-100.hdf5", "r") as f:
        result = f["wasserstein_distance"]

        parameters = dict(zip(result.attrs.keys(), result.attrs.values()))
        ws_distance = np.array(result["wasserstein_distance"])
        t = np.array(result["t"])

    return t, ws_distance, parameters


def test_network_creation():
    path = "/home/manuel/Documents/code/SpoNetResearch/juliacode/ER_n10_p-crit-10"
    print(read_network(path))


def main():
    return


if __name__ == "__main__":
    print(read_run())