from sponet import CNVMParameters
import numpy as np


def cnvm_3s_asymm(network_gen):

	R = np.array([[0, 0.8, 0.2],
	             [0.2, 0, 0.8],
	             [0.8, 0.2, 0]])
	Rt = np.array([[0, 0.01, 0.01],
	              [0.01, 0, 0.01],
	              [0.01, 0.01, 0]])

	x_init_shares = [0.2, 0.5, 0.3]

	params = CNVMParameters(
		num_opinions=3,
		network_generator=network_gen,
		r=R,
		r_tilde=Rt,
		alpha=1
	)

	return params, x_init_shares, "CNVM_3s_asymm"
