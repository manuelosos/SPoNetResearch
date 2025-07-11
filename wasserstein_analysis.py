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
	edge_prob_names = {}
	param_combs = []
	for params in available_network_params:
		n_nodes.append(params['n_nodes'])
		edge_probs.append(params['edge_probability'])
		edge_prob_names[params["edge_probability"]] = params['edge_probability_name']

		param_combs.append((params["n_nodes"], params["edge_probability"], params["edge_probability_name"]))

	n_nodes_uniq = set(sorted(n_nodes))
	edge_probs_uniq = set(sorted(edge_probs))

	missing =[]
	for n in n_nodes_uniq:
		for edge_prob in edge_probs_uniq:
			if np.log(n)/n > edge_prob:
				continue
			tmp = (n, edge_prob, edge_prob_names[edge_prob])
			if tmp not in param_combs:
				missing.append(tmp)


	print(missing)
	for miss in missing:
		print(int(miss[0]), miss[2])


	plt.scatter(n_nodes, edge_probs, marker="x")
	plt.yscale("log")
	plt.xscale("log")
	plt.show()





if __name__ == '__main__':
	show_available_network_params("/home/manuel/Documents/code/data/networks/networks/")
