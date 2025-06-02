import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix

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

    res = np.load("ER_p20_N12.npz")
    print(res.keys())
    print(res["edge_density"])

def main():
    return




def load_sparse_matrix_npz(filename):
    """
    Load a sparse matrix saved by Julia as a .npz file.
    Returns a scipy.sparse.csc_matrix.
    """
    loader = np.load(filename)
    data = loader["data"]
    indices = loader["indices"]
    indptr = loader["indptr"]-1
    shape = tuple(loader["shape"])

    print(indptr)
    return csc_matrix((data, indices, indptr), shape=shape)


if __name__ == "__main__":
    # plot_wasserstein_distance_vs_network_size()
    res=load_sparse_matrix_npz("/home/manuel/Documents/code/SpoNetResearch/juliacode/ER_p-crit1000_N1000.npz")
    print(res)