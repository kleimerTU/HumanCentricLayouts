import argparse
import pickle
import sys
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from src import utils
from src.main_functions import *

def main(argv):
    parser = argparse.ArgumentParser(
        description="Create and evaluate scenes for every trained epoch of the given model (without collision detection)"
    )

    parser.add_argument(
        "path_to_config",
        nargs='+',
        type=str,
        help="Paths to the configuration files (accepts multiple inputs)"
    )
    
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=200,
        help="Maximum number of sequences to process in parallel (requires more GPU memory)"
    )
    
    parser.add_argument(
        "--no_synthesis",
        action="store_true",
        help="If set, no new scenes are generated and only saved data is plotted"
    )
    
    parser.add_argument(
        "--manual_seed",
        type=int,
        default=None,
        help="Sets a manual seed for the random sampling"
    )
    
    fig = plt.figure(figsize=(20,4))
    ax_train = plt.subplot(131)
    ax_val = plt.subplot(132)
    ax_ergo = plt.subplot(133)
    
    args = parser.parse_args(argv)
    if args.manual_seed:
      torch.manual_seed(args.manual_seed)
    max_batch_size = args.max_batch_size
    n_models = len(args.path_to_config)
    legend_names = []
    for i in range(n_models):
        with open(args.path_to_config[i], "r") as f:
          config = yaml.load(f, Loader=Loader)
        
        n_scenes = config["synthesis"]["n_scenes"]
        top_p = config["synthesis"]["top_p"]
        top_k = config["synthesis"]["top_k"]
        max_noncat = config["synthesis"]["max_noncat"]
        max_windoor = config["synthesis"]["max_windoor"]
        set_name = config["general"]["set_name"]
        suffix_name = config["general"]["suffix_name"]
        max_furniture = config["general"]["max_furniture"]
        res = config["general"]["res"]
        max_seq_len = (max_furniture + 1) * 6 + 1
        order_switch = config["general"]["order_switch"]
        if torch.cuda.is_available():
          device = torch.device(config["network"]["device"])
        else:
          device = torch.device("cpu")
        path_input_data = config["paths"]["path_input_data"] + "/"
        path_output_data = config["paths"]["path_output_data"] + "/sequence/"
        path_trained_models = config["paths"]["path_trained_models"] + "/"
        n_versions = config["network"]["epochs"]
        model_names = [config["network"]["model_name"]]
        legend_names.append(config["network"]["model_name"])
        use_alt_loss = config["network"]["use_alt_loss"]
        category_valid = utils.get_valid_categories()
        _, _, dict_cat2fun = utils.get_category_dicts(res)
        minvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_minvalues" + suffix_name + ".pkl", "rb" ))
        maxvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_maxvalues" + suffix_name + ".pkl", "rb" ))
        
        if args.no_synthesis:
          sampling_type = utils.get_sampling_type(top_k,top_p,max_noncat,max_windoor)
        else:
          with torch.no_grad():
            if not os.path.isdir(path_output_data):
              os.mkdir(path_output_data)
            sampling_type = generate_and_rank_scenes(n_scenes, n_versions, model_names, minvalue_dict, maxvalue_dict, dict_cat2fun, 
              path_output_data, path_trained_models, res=res, max_seq_len=max_seq_len, top_k=top_k, top_p=top_p, order_switch=order_switch,
              max_noncat=max_noncat, max_windoor=max_windoor, max_batch_size=max_batch_size,use_alt_loss=use_alt_loss,device=device)

        #ax = plt.subplot(131)
        for model_name in model_names:
          model_name = 'model_' + model_name
          train_loss_hist = pickle.load(open(path_trained_models + model_name + "/model_tmp" + str(n_versions-1) + "/train_loss_hist.pkl", "rb" ))
          ax_train.plot(train_loss_hist)

        #ax = plt.subplot(132)
        for model_name in model_names:
          model_name = 'model_' + model_name
          train_loss_hist = pickle.load(open(path_trained_models + model_name + "/model_tmp" + str(n_versions-1) + "/val_loss_hist.pkl", "rb" ))
          ax_val.plot(train_loss_hist)

        #ax = plt.subplot(133)
        for model_name in model_names:
          model_name = 'model_' + model_name
          version_mean_scores = pickle.load(open(path_output_data + model_name + "/" + sampling_type + "_mean_scores.pkl", "rb" ))
          ax_ergo.plot(version_mean_scores)
          
    ax_train.legend(legend_names)
    ax_train.title.set_text('Training loss')      
    ax_val.legend(legend_names)
    ax_val.title.set_text('Validation loss')
    ax_ergo.legend(legend_names)
    ax_ergo.title.set_text('Mean ergo score of generated scenes with ' + sampling_type + ' sampling')
    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1:])