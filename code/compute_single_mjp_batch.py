
from utils.computation_utils import *
from utils.network_utils import *
from utils.parameter_utils import *
import os
import argparse


parser = argparse.ArgumentParser(
    description="Compute single MJP batch."
)

parser.add_argument(
    "rate_type",
    type=str,
    help="Keyword for the specific transition rates and initial values."
)
parser.add_argument(
    "n_states",
    type=int,
    help="Number of states."
)
parser.add_argument(
    "network_path",
    type=str,
    help="Path to the network that is used for simulation."
)
parser.add_argument(
    "batch_id",
    type=str,
    help="Batch id used to identify different batches and avoid overwriting."
)


def compute_single_mjp_batch(
        ws_params: WassersteinParameters,
        save_path: str | os.PathLike,
        batch_id: str
):
    save_dir_path = os.path.join(save_path, ws_params.run_name)
    if not os.path.isdir(save_dir_path):
        os.mkdir(save_dir_path)

    cv = OpinionShares(ws_params.n_states, normalize=True)
    t, traj = compute_mjp_batch(
        ws_params,
        cv
    )

    batch_name = f"mjp_batch_{batch_id}.npz"
    save_path = os.path.join(save_dir_path, batch_name)

    save_batch(
        save_path,
        time_traj=t,
        trajectories=traj
    )

    return


if __name__ == "__main__":

    args = parser.parse_args()
    ws_params = standard_ws_from_network_and_rate_type(args.n_states, args.rate_type, args.network_path)