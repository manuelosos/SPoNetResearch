import os
import argparse
from network_utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
	"--test",
	action="store_true",
	default=False,
	help="Test mode."
)



def create_cluster_skripts(
	path: str,
	test: bool = True
):

	network_names = get_available_networks(path)

	for network_name in network_names:
		print(get_network_params_from_name(network_name))



	print("test")
	return


if __name__ == "__main__":
	args = parser.parse_args()
	create_cluster_skripts(test=args.test)