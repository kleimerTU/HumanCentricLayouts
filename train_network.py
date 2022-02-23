import argparse
import sys
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from src.main_functions import *
from src.network import *

def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a Transformer network with the parameters specified in the config file. To train more than one model, multiple config files can be specified."
    )

    parser.add_argument(
        "path_to_config",
        nargs='+',
        type=str,
        help="Paths to the configuration files (accepts multiple inputs)"
    )
    
    args = parser.parse_args(argv)
    config_paths = args.path_to_config
    n_models = len(config_paths)
    for i in range(n_models):
      with open(args.path_to_config[i], "r") as f:
        config = yaml.load(f, Loader=Loader)
    
      training_params = dict(config["general"], **config["network"])
      training_params["path_input_data"] = config["paths"]["path_input_data"] + "/"
      training_params["path_trained_models"] = config["paths"]["path_trained_models"] + "/"
        
      training_loop(training_params)
    
if __name__ == "__main__":
    main(sys.argv[1:])