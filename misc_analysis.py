import numpy as np
from fontTools.mtiLib import parseLookupRecords

from pythoncode.utils.computation_utils import compute_single_mjp_sde_batch
from pythoncode.utils.parameter_utils import WassersteinParameters, standard_ws_with_new_network_from_rate_type
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from itertools import product


def simplex_embedding(x):
    """
    Map from the n-simplex in R^n to R^{n-1}, e.g. from 3D simplex to 2D plane
    """
    n = len(x)
    angles = 2 * np.pi * np.arange(n) / n
    vertices = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # shape (n, 2)
    return x @ vertices  # linear combination (barycentric coordinates)


def dist_plot(ax, traj, coord1, coord2):


	sns.kdeplot(
		ax=ax, x=traj[:, :, coord1].flatten(), y=traj[:, :, coord2].flatten(),
		cmap="mako", fill=True
	)

	return



def complete_dist_plot(traj):

	n_states = traj.shape[-1]

	fig, axes = plt.subplots(n_states, n_states)

	plot_args = []

	for i, j in product(range(n_states), range(n_states)):
		plot_args.append((axes[i, j], traj, i, j))

	with mp.Pool(processes=mp.cpu_count()) as pool:
		pool.starmap(dist_plot, plot_args)




def main():

	n_nodes = 100
	edge_probability = 0.5

	n_states = 3
	rate_type = "asymm"
	n_runs = 1
	batch_size = n_runs


	comp_params = standard_ws_with_new_network_from_rate_type(
		n_nodes,
		edge_probability,
		n_states,
		rate_type,
		batchsize_mjp=batch_size,
		batchsize_sde=batch_size,
		n_runs=n_runs,
	)
	comp_params.t_max = 200
	comp_params. save_resolution = 5



	t_mjp, x_mjp, t_sde, x_sde = compute_single_mjp_sde_batch(comp_params)

	print("Finished simulation")

	x_mjp_emb = np.apply_along_axis(simplex_embedding, 2, x_mjp)
	x_sde_emb = np.apply_along_axis(simplex_embedding, 2, x_sde)

	print("Finished embedding")

	fig, axes = plt.subplots(1, 2, constrained_layout=True)

	coord1 = 0
	coord2 = 1

	ax_mjp, ax_sde = axes

	for ax in axes.flatten():
		ax.set_aspect("equal")
		ax.set_xlim(-0.5, 0.8)
		ax.set_ylim(-0.75, 0.75)


	ax_sde.set_title("SDE")
	ax_mjp.set_title("MJP")


	dist_plot(ax_mjp, x_mjp_emb, coord1, coord2)
	dist_plot(ax_sde, x_sde_emb, coord1, coord2)


	ax_mjp.scatter(x_mjp_emb[0, 0, 0], x_mjp_emb[0, 0, 1], marker="x")
	ax_sde.scatter(x_sde_emb[0, 0, 0], x_sde_emb[0, 0, 1], marker="x")









	plt.show()














	return


if __name__ == "__main__":
	main()

