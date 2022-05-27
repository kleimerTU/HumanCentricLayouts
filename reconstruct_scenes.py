import argparse
import json
import pickle
import sys
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from src import utils
from src.main_functions import *
from src.reconstruction import *
from src.modeling_gpt2_custom import GPT2LMHeadModelCustom

def main(argv):
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D scenes from the generated sequences"
    )

    parser.add_argument(
        "path_to_config",
        nargs='+',
        type=str,
        help="Paths to the configuration files (accepts multiple inputs)"
    )
    
    parser.add_argument(
        "--with_floors",
        action="store_true",
        help="If true, loads the floors from the validation set for scenes conditioned on rooms"
    )
    
    args = parser.parse_args(argv)
    n_models = len(args.path_to_config)
    for i in range(n_models):
        with open(args.path_to_config[i], "r") as f:
          config = yaml.load(f, Loader=Loader)
        
        n_sequences = config["synthesis"]["n_scenes"]
        top_p = config["synthesis"]["top_p"]
        top_k = config["synthesis"]["top_k"]
        max_noncat = config["synthesis"]["max_noncat"]
        max_windoor = config["synthesis"]["max_windoor"]
        sampling_type = utils.get_sampling_type(top_k,top_p,max_noncat,max_windoor)
        epoch = config["synthesis"]["recon_epoch"]
        house_name = config["synthesis"]["house_name"]
        set_name = config["general"]["set_name"]
        suffix_name = config["general"]["suffix_name"]
        max_furniture = config["general"]["max_furniture"]
        res = config["general"]["res"]
        max_seq_len = (max_furniture + 1) * 6 + 1
        order_switch = config["general"]["order_switch"]
        path_input_data = config["paths"]["path_input_data"] + "/"
        path_output_data = config["paths"]["path_output_data"] + "/"
        path_trained_models = config["paths"]["path_trained_models"] + "/"
        path_3dfront_data = config["paths"]["path_3dfront_data"] + "/"
        n_versions = config["network"]["epochs"]
        model_name = "model_" + config["network"]["model_name"]
        category_valid = utils.get_valid_categories()
        dict_cat2int, dict_int2cat, dict_cat2fun = utils.get_category_dicts(res)
        minvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_minvalues" + suffix_name + ".pkl", "rb" ))
        maxvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_maxvalues" + suffix_name + ".pkl", "rb" ))
        
        dict_cat2model = pickle.load(open(path_3dfront_data + "dict_cat2model_ext.pkl", "rb"))
        dict_cat2dims = pickle.load(open(path_3dfront_data + "dict_cat2dims_ext.pkl", "rb"))
        invalid_models = []
        with open(path_3dfront_data + "model2minp.json") as file:
          model2minp = json.load(file)
        with open(path_3dfront_data + "model2maxp.json") as file:
          model2maxp = json.load(file)
        with open(path_3dfront_data + "invalid_models.txt") as file:
          invalid_model_lines = file.readlines()
          invalid_models.extend([line.rstrip() for line in invalid_model_lines])
          
        sequences = pickle.load( open(path_output_data + "sequence/" + model_name + "/resampled_" + sampling_type + "_sequences.pkl", "rb" ))
        scores = pickle.load( open(path_output_data + "sequence/" + model_name + "/resampled_" + sampling_type + "_scores.pkl", "rb" ))
        indices = torch.linspace(0,sequences.size(0)-1,n_sequences,dtype=torch.long)
        
        cat_supporter = ['table','desk','coffee_table','side_table','dresser','dressing_table','sideboard','shelving','stand','tv_stand']
        cat_supported = ['television','computer','indoor_lamp']
        
        if args.with_floors:
          val_floors = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_val_floors" + suffix_name + ".pkl", "rb" ))
          seq_indices = pickle.load(open( path_output_data + "sequence/" + model_name + "/resampled_" + sampling_type + "_seq_indices.pkl", "rb" ))
            
        nadded = 0
        for j in range(n_sequences):   
          print("Reconstructing scenes of model", config["network"]["model_name"], "-", str(j+1), "of" , str(n_sequences) , end="\r")
          if nadded == n_sequences:
            break
          i = indices[j].item()
          full_house_name = house_name + str(nadded)
          room_seq = sequences[[i],:-1]
          nodes, has_coll, scene_dict, furniture = add_furniture(room_seq, minvalue_dict, maxvalue_dict, dict_int2cat, dict_cat2model, dict_cat2dims,
            model2minp, model2maxp, res=res, invalid_models=invalid_models, order_switch=0, add_all=True, collision_check=False)
          
          if args.with_floors:
            augment = seq_indices[i] % 8
            floor = val_floors[seq_indices[i]]
            reconstruct_3d_scene(full_house_name, nodes, scene_dict, furniture, cat_supporter, cat_supported, path_output_data, 
            floor_mesh=floor, augment=augment)
          else:
            reconstruct_3d_scene(full_house_name, nodes, scene_dict, furniture, cat_supporter, cat_supported, path_output_data)
          nadded = nadded + 1
          
if __name__ == "__main__":
    main(sys.argv[1:])