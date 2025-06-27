from typing import List, Dict, Any
from collections import namedtuple
import numpy as np
import scipy as sp
import datetime
import os
import h5py

from pythoncode.utils.parameter_utils import WassersteinParameters


def load_batches(paths_batches: List[str]) -> tuple[np.ndarray, np.ndarray]:

	res = np.load(paths_batches[0])["x"]
	t = np.load(paths_batches[0])["t"]

	for path_batch in paths_batches[1:]:
		res = np.concatenate((res, np.load(path_batch)["x"]), axis=0)
	return t, res


def compute_wasserstein_distance_from_batches(
		paths_batches_mjp: List[str],
		paths_batches_sde: List[str],
		verbose=False
):

	t, trajectories_mjp = load_batches(paths_batches_mjp)
	t, trajectories_sde = load_batches(paths_batches_sde)

	n_time_steps = trajectories_mjp.shape[1]
	n_states = trajectories_mjp.shape[2]

	distances_wasserstein = np.empty((n_time_steps, n_states))

	if verbose:
		print("Starting Wasserstein distance computation...")

	for time_step in range(n_time_steps):
		for state in range(n_states):

			distances_wasserstein[time_step, state] = (
				sp.stats.wasserstein_distance(
					trajectories_mjp[:, time_step, state],
					trajectories_sde[:, time_step, state]
				)
			)
	if verbose:
		print("Wasserstein distance computation complete.")

	return t, distances_wasserstein


def save_wasserstein_result(
		path: str | os.PathLike,
		ws_params: WassersteinParameters,
		error_trajectory: np.ndarray,
		time_trajectory: np.ndarray
):
	if not path.endswith(".hdf5"):
		path += ".hdf5"


	with h5py.File(path, "w") as save_file:
		ws_dist_group = save_file.create_group("wasserstein_distance")

		ws_dist_group.attrs["rate_type"] = ws_params.rate_type
		ws_dist_group.attrs["n_states"] = ws_params.n_states
		for attribute in ws_params.network_params.keys():
			ws_dist_group.attrs[attribute] = ws_params.network_params[attribute]
		ws_dist_group.attrs["params_str"] = str(ws_params.cnvm_params)
		ws_dist_group.attrs["n_runs_sde"] = ws_params.n_runs_sde
		ws_dist_group.attrs["n_runs_mjp"] = ws_params.n_runs_mjp
		ws_dist_group.attrs["sde_simulation_resolution"] = ws_params.simulation_resolution_sde
		ws_dist_group.attrs["date_of_computation"] = datetime.datetime.now().isoformat()

		ws_dist_group.create_dataset("wasserstein_distance", data=error_trajectory)
		ws_dist_group.create_dataset("t", data=time_trajectory)

	return



# Optional: a lightweight return type
WassersteinResult = namedtuple(
	"WassersteinResult",
	["attrs", "error_trajectory", "time_trajectory"],
)


def load_wasserstein_result(path: str | os.PathLike) -> WassersteinResult:
	# Ensure the expected file extension
	path = os.fspath(path)
	if not path.endswith(".hdf5"):
		path += ".hdf5"

	# Open read-only and extract everything you need
	with h5py.File(path, "r") as h5f:
		g = h5f["wasserstein_distance"]

		# Copy attributes into a plain Python dict (so the file can close safely)
		attrs: Dict[str, Any] = {k: g.attrs[k] for k in g.attrs.keys()}

		# Datasets -> NumPy arrays
		error_trajectory: np.ndarray = g["wasserstein_distance"][()]
		time_trajectory:  np.ndarray = g["t"][()]

	return WassersteinResult(attrs, error_trajectory, time_trajectory)

