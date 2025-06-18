from typing import List
import numpy as np
from dataclasses import dataclass
from .network_utils import create_equal_network_init_and_shares
from sponet.collective_variables import OpinionShares
from sponet import sample_many_runs, CNVM, CNVMParameters, sample_cle
import os
import networkx as nx
from .parameter_utils import *




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
		overwrite: bool=False
):
	if not overwrite and os.path.exists(save_path):
		np.savez_compressed(save_path, t=time_traj, x=trajectories)

	return


def compute_mjp_batch(
		comp_params: WassersteinParameters,
		cv
):

	t, x = sample_many_runs(
		params=comp_params.cnvm_params,
		initial_states=np.array(comp_params.network_init),
		t_max=comp_params.t_max,
		num_timesteps=comp_params.save_resolution * comp_params.t_max + 1,
		num_runs=comp_params.batchsize_mjp,
		collective_variable=cv
	)

	return t, x


def compute_sde_batch(
		cnvm_params: CNVMParameters,
		comp_params: WassersteinParameters,
		initial_shares: np.ndarray,
):

	t, x = sample_cle(
		params=cnvm_params,
		initial_state=initial_shares,
		max_time=comp_params.t_max,
		num_time_steps=comp_params.t_max * comp_params.simulation_resolution_sde * comp_params.save_resolution,
		num_samples=comp_params.batchsize_sde,
		saving_offset=comp_params.simulation_resolution_sde
	)

	return t, x


def compute_mjp_sde_runs(
		comp_params: WassersteinParameters,
		batch_save_path: str = "",
		verbose = True,
		batch_id: str = "",
		overwrite: bool = False
) -> tuple[List[str], List[str]]:

	cv = OpinionShares(comp_params.n_states, normalize=True)

	x_init_shares = cv(np.array([comp_params.init_network]))

	batches_mjp = get_batchsizes(comp_params.n_runs_mjp, comp_params.batchsize_mjp)
	paths_batches_mjp = []

	print(f"Starting MJP simulation for {comp_params.n_runs_mjp} in {len(batches_mjp)} batches with batch id {batch_id}.")
	for i, n_runs in enumerate(batches_mjp):

		t, x = compute_mjp_batch(
			comp_params=comp_params,
			cv=cv
		)

		batch_save_path = os.path.join(batch_save_path, f"{batch_id}_batch_{i}_mjp.npz")
		paths_batches_mjp.append(batch_save_path)

		save_batch(batch_save_path, t, x[0], overwrite)

		if verbose:
			print(f"Finished batch {i} with {n_runs} and batch id {batch_id} runs of MJP simulation.")

	if verbose:
		print("Finished MJP simulation.")


	batches_sde = get_batchsizes(comp_params.n_runs_sde, comp_params.batchsize_sde)
	paths_batches_sde = []

	print(f"Starting SDE simulation for {comp_params.n_runs_sde} in {len(batches_sde)} batches with batch_id {batch_id}.")
	for i, n_runs in enumerate(batches_sde):

		t, x = compute_sde_batch(
			cnvm_params=comp_params.cnvm_params,
			comp_params=comp_params,
			initial_shares=x_init_shares[0]
		)

		batch_save_path = os.path.join(batch_save_path, f"{batch_id}_batch_{i}_sde.npz")
		paths_batches_sde.append(batch_save_path)

		save_batch(batch_save_path, t, x, overwrite)

		if verbose:
			print(f"Finished batch {i} with {n_runs} runs and batch id {batch_id} of SDE simulation.")

	if verbose:
		print("Finished SDE simulation.")

	return paths_batches_mjp, paths_batches_sde
