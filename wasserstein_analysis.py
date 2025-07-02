import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp

from pythoncode.utils.network_utils import get_available_network_params, read_network


def show_available_network_params(path: str):
	available_network_params = get_available_network_params(path)

	edge_probs = []
	n_nodes = []
	for params in available_network_params:
		n_nodes.append(params['n_nodes'])
		edge_probs.append(params['edge_probability'])

	plt.scatter(n_nodes, edge_probs, marker="x")
	plt.yscale("log")
	plt.xscale("log")
	plt.show()





if __name__ == '__main__':
	show_available_network_params("/home/manuel/Documents/code/data/networks/networks/")
