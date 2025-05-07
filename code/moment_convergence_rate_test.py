import argparse
import json
import numpy as np
from sponet import sample_moments

parser = argparse.ArgumentParser(
    description="Test for the convergence rate of moments."
                "Not to be confused with convergence in moment"
)
parser.add_argument("--test", action="store_true", help="Set to test the script")

with open("paths.json") as file:
    data = json.load(file)
data_path = data["data_path"]



def main(test = False):


    R = np.array([[0, .8, .2],
                  [.2, 0, .9],
                  [.8, .3, 0]])

    Rt = 0.01 * np.array([[0, .9, .8],
                          [.7, 0, .9],
                          [.9, .7, 0]])
    n_states = R.shape[0]

    if test:
        t_max = 10
        n_runs = 10
        n_nodes_list = [10]
    else:
        t_max = 50
        n_runs = 1000000
        n_nodes_list = [10, 100, 1000, 10000, 100000, 1000000]

    errs = []

    for n_nodes in n_nodes_list:
        x_init_shares, x_init_network = (
            create_equal_network_init_and_shares([.2, .5, .3], n_nodes))

        params = CNVMParameters(num_opinions=n_states,
                                num_agents=n_nodes,
                                r=R,
                                r_tilde=Rt,
                                alpha=1)
        diffs = compare_wasserstein(params, x_init_network, n_runs, t_max, verbose=True)
        errs.append(diffs)

        if test:
            plt.plot(diffs)
            plt.show()

    return

if __name__ == "__main__":
    main()