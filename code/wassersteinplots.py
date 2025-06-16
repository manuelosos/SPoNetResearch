import json
import h5py
import os.path as osp
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
import scipy as sp
from utils.network_utils import get_available_networks, read_network

with open("paths.json") as file:
    data = json.load(file)
data_path = data.get("data_path")


def read_wasserstein_results(
        path: str,
):

    dir_list = os.listdir(path)

    t = None
    trajectories = []
    parameter_list = []

    for entry in dir_list:
        if osp.isdir(osp.join(path, entry)) and (entry+".hdf5") in os.listdir(osp.join(path, entry)):
            t, trajectory, parameters = read_run(osp.join(path, entry+".hdf5"))
            parameter_list.append(parameters)
            trajectories.append(trajectory)

    return t, trajectories, parameter_list


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


def read_run(
        path: str
):
    
    with h5py.File(path, "r") as f:
        result = f["wasserstein_distance"]

        parameters = dict(zip(result.attrs.keys(), result.attrs.values()))
        ws_distance = np.array(result["wasserstein_distance"])
        t = np.array(result["t"])

    return t, ws_distance, parameters


if __name__ == "__main__":
    pass
