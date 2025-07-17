import json
import h5py
import os.path as osp
import os
import numpy as np
import matplotlib.pyplot as plt


with open("../paths.json") as file:
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
            t, trajectory, parameters = read_run(osp.join(path,entry, entry+".hdf5"))
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


def plot_wasserstein_distance_vs_n_nodes_and_edge_probability():

    path = "/home/manuel/Documents/code/data/ws_distance/ws_distance/asymm_3/"

    t, trajectories, parameter_list = read_wasserstein_results(path)


    unique_edge_probability_names = np.unique(np.array([params["edge_probability_name"].decode() for params in parameter_list]))


    plot_trajs = dict()
    for edge_prob_name in unique_edge_probability_names:
        plot_trajs[edge_prob_name] = (list(), list())



    for traj, params in zip(trajectories, parameter_list):

        edge_prob_name = params["edge_probability_name"].decode()
        plot_trajs[edge_prob_name][0].append(np.linalg.norm(traj, np.inf))
        plot_trajs[edge_prob_name][1].append(params["n_nodes"])


    error_bound_proven = lambda x: np.log(x)/np.sqrt(x)
    error_bound_fitting = lambda x: np.log(x)/x


    x_err_bound = np.linspace(10, 50_000)



    fig, ax = plt.subplots(constrained_layout=True)

    ax.plot(x_err_bound, error_bound_proven(x_err_bound) ,label=r"$\frac{\log(N)}{\sqrt{N}}$")
    ax.plot(x_err_bound[:3], error_bound_fitting(x_err_bound)[:3] ,label=r"$\frac{\log(N)}{\sqrt{N}}$")

    for edge_prob_name in unique_edge_probability_names:
        traj = np.array(plot_trajs[edge_prob_name][0])
        n_nodes = np.array(plot_trajs[edge_prob_name][1])
        sorting = np.argsort(n_nodes)


        ax.plot(n_nodes[sorting], traj[sorting], label=edge_prob_name, marker="x", lw=0.75)

    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.legend(ncol=3)

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
    plot_wasserstein_distance_vs_n_nodes_and_edge_probability()
