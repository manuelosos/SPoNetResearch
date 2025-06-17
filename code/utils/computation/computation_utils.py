from typing import List
import numpy as np
from dataclasses import dataclass
from sponet.collective_variables import OpinionShares
from sponet import sample_many_runs, CNVM, CNVMParameters, sample_cle
import os
import networkx as nx


@dataclass
class WassersteinParameters:
	n_states: int
	rate_type: str
	network: nx.Graph
	network_params: dict
	t_max: int
	n_runs_mjp: int
	n_runs_sde: int
	batchsize_mjp: int
	batchsize_sde: int
	save_resolution: int
	simulation_resolution_sde: int


def get_run_name(
		comp_params: WassersteinParameters
) -> str:
	run_name = f"ws_dist_{comp_params.n_states}s_{comp_params.rate_type}_{comp_params.network_params['network_name'].decode()}"
	return run_name



def get_batchsizes(
		n_total_runs: int,
		batchsize: int
):

	n_full_batches = n_total_runs // batchsize
	n_runs_partial_batch = n_total_runs % batchsize

	batches = n_full_batches * [batchsize]
	if n_runs_partial_batch > 0:
		batches += [n_runs_partial_batch]

	return batches


def save_batch(
		save_path: str,
		time_traj: np.ndarray,
		trajectories: np.ndarray,
):
	np.savez_compressed(save_path, t=time_traj, x=trajectories)

	return


def compute_mjp_batch(
		params: CNVMParameters,
		initial_states,
		t_max,
		n_runs,
		save_resolution,
		cv
):

	t, x = sample_many_runs(
		params=params,
		initial_states=np.array(initial_states),
		t_max=t_max,
		num_timesteps=save_resolution * t_max + 1,
		num_runs=n_runs,
		collective_variable=cv
	)


	return


def compute_mjp_sde_runs(
		comp_params: WassersteinParameters,
		cnvm_params: CNVMParameters,
		x_init_network: np.ndarray,
		batch_save_path: str = "",
		verbose = True,
		batch_id: str = ""
) -> tuple[List[str], List[str]]:
	"""
    Computes trajectories of a Markov jump process on a network and the corresponding diffusion approximation.
    :param comp_params:
    :param cnvm_params:
        CNVM parameters for the jump process model.
    :param x_init_network:
        Initial network state. This initial state is used for all computations.
    :param batch_save_path:
        Path to save the results of the batches.
    :param verbose:
    :param batch_id:
    :return:
	"""
	cv = OpinionShares(cnvm_params.num_opinions, normalize=True)


	x_init_shares = cv(np.array([x_init_network]))

	batches_mjp = get_batchsizes(comp_params.n_runs_mjp, comp_params.batchsize_mjp)
	paths_batches_mjp = []

	print(f"Starting MJP simulation for {comp_params.n_runs_mjp} in {len(batches_mjp)} batches with batch id {batch_id}.")
	for i, n_runs in enumerate(batches_mjp):

		t, x = sample_many_runs(
			params=cnvm_params,
			initial_states=np.array([x_init_network]),
			t_max=comp_params.t_max,
			num_timesteps=comp_params.save_resolution*comp_params.t_max+1,
			num_runs=n_runs,
			collective_variable=cv
		)

		path_batch = os.path.join(batch_save_path, f"{batch_id}_batch_{i}_mjp.npz")
		paths_batches_mjp.append(path_batch)

		np.savez_compressed(path_batch, t=t, x=x[0])

		if verbose:
			print(f"Finished batch {i} with {n_runs} and batch id {batch_id} runs of MJP simulation.")

	if verbose:
		print("Finished MJP simulation.")


	batches_sde = get_batchsizes(comp_params.n_runs_sde, comp_params.batchsize_sde)
	paths_batches_sde = []

	print(f"Starting SDE simulation for {comp_params.n_runs_sde} in {len(batches_sde)} batches with batch_id {batch_id}.")
	for i, n_runs in enumerate(batches_sde):

		t, x = sample_cle(
			params=cnvm_params,
			initial_state=x_init_shares[0],
			max_time=comp_params.t_max,
			num_time_steps=comp_params.t_max*comp_params.simulation_resolution_sde*comp_params.save_resolution,
			num_samples=n_runs,
			saving_offset=comp_params.simulation_resolution_sde
		)

		path_batch = os.path.join(batch_save_path, f"{batch_id}_batch_{i}_sde.npz")
		paths_batches_sde.append(path_batch)

		np.savez_compressed(path_batch, t=t, x=x)

		if verbose:
			print(f"Finished batch {i} with {n_runs} runs and batch id {batch_id} of SDE simulation.")

	if verbose:
		print("Finished SDE simulation.")

	return paths_batches_mjp, paths_batches_sde
