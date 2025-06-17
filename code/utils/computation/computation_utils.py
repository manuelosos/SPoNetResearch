from typing import List

import sponet
import logging
import numpy as np
from logging import log
import sponet
from sponet.network_generator import ErdosRenyiGenerator
from sponet.collective_variables import OpinionShares
from sponet import sample_many_runs, CNVM, CNVMParameters, sample_cle
import os


def get_batches(n_total_runs: int, batchsize: int):

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
		params: CNVMParameters,
		x_init_network: np.ndarray,
		n_runs_sde: int,
		n_runs_mjp: int,
		t_max: int,
		save_resolution = 2,
		simulation_resolution_sde = 20,
		batchsize_sde = 10000,
		batchsize_mjp = 1000,
		save_path_batch: str = "",
		verbose = True,
		batch_id: str = ""
) -> tuple[List[str], List[str]]:
	"""
	Computes trajectories of a Markov jump process on a network and the corresponding diffusion approximation.
	:rtype Tuple[List[str], List[str]]
	:param params:
		CNVM parameters for the jump process model.
	:param x_init_network:
		Initial network state. This initial state is used for all computations.
	:param n_runs_sde:
	:param n_runs_mjp:
	:param t_max:
	:param save_resolution:
		Number of timesteps to save per time unit.
		t_max=2 and save_resolution=2 will save 4 timesteps in total.
	:param simulation_resolution_sde:
		Number of timesteps simulated per time unit for Euler Maruyama simulation of the SDE.
	:param batchsize_sde:
	:param batchsize_mjp:
	:param save_path_batch:
		Path to save the results of the batches.
	:param verbose:
	:param batch_id:
	:return:
	"""
	cv = OpinionShares(params.num_opinions, normalize=True)
	x_init_shares = cv(np.array([x_init_network]))

	batches_mjp = get_batches(n_runs_mjp, batchsize_mjp)
	paths_batches_mjp = []

	print(f"Starting MJP simulation for {n_runs_mjp} in {len(batches_mjp)} batches with batch id {batch_id}.")
	for i, n_runs in enumerate(batches_mjp):

		t, x = sample_many_runs(
			params=params,
			initial_states=np.array([x_init_network]),
			t_max=t_max,
			num_timesteps=save_resolution*t_max+1,
			num_runs=n_runs,
			collective_variable=cv
		)

		path_batch = os.path.join(save_path_batch, f"{batch_id}_batch_{i}_mjp.npz")
		paths_batches_mjp.append(path_batch)
		np.savez_compressed(path_batch, t=t, x=x[0])

		if verbose:
			print(f"Finished batch {i} with {n_runs} and batch id {batch_id} runs of MJP simulation.")

	if verbose:
		print("Finished MJP simulation.")


	batches_sde = get_batches(n_runs_sde, batchsize_sde)
	paths_batches_sde = []

	print(f"Starting SDE simulation for {n_runs_sde} in {len(batches_sde)} batches with batch_id {batch_id}.")
	for i, n_runs in enumerate(batches_sde):

		t, x = sample_cle(
			params=params,
			initial_state=x_init_shares[0],
			max_time=t_max,
			num_time_steps=t_max*simulation_resolution_sde*save_resolution,
			num_samples=n_runs,
			saving_offset=simulation_resolution_sde
		)

		path_batch = os.path.join(save_path_batch, f"{batch_id}_batch_{i}_sde.npz")
		paths_batches_sde.append(path_batch)
		np.savez_compressed(path_batch, t=t, x=x)

		if verbose:
			print(f"Finished batch {i} with {n_runs} runs and batch id {batch_id} of SDE simulation.")

	if verbose:
		print("Finished SDE simulation.")

	return paths_batches_mjp, paths_batches_sde
