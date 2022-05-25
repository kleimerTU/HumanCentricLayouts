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
      description="Create sequences for the given model and epoch (with collision detection if enabled)"
  )

  parser.add_argument(
      "path_to_config",
      nargs='+',
      type=str,
      help="Paths to the configuration files (accepts multiple inputs)"
  )
  
  parser.add_argument(
      "--num_room_plots",
      default=0,
      type=int,
      help="Optionally plots the specified number of rooms and saves them in the output data directory"
  )
  
  parser.add_argument(
      "--manual_seed",
      type=int,
      default=None,
      help="Sets a manual seed for the random sampling"
  )
  
  args = parser.parse_args(argv)
  if args.manual_seed:
    torch.manual_seed(args.manual_seed)
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
    collision_check = config["synthesis"]["collision_check"]
    epoch = config["synthesis"]["recon_epoch"]
    house_name = config["synthesis"]["house_name"]
    set_name = config["general"]["set_name"]
    suffix_name = config["general"]["suffix_name"]
    max_furniture = config["general"]["max_furniture"]
    res = config["general"]["res"]
    max_seq_len = (max_furniture + 1) * 6 + 1
    order_switch = config["general"]["order_switch"]
    path_input_data = config["paths"]["path_input_data"] + "/"
    path_output_data = config["paths"]["path_output_data"] + "/sequence/"
    path_trained_models = config["paths"]["path_trained_models"] + "/"
    path_3dfront_data = config["paths"]["path_3dfront_data"] + "/"
    n_versions = config["network"]["epochs"]
    model_names = [config["network"]["model_name"]]
    if torch.cuda.is_available():
      device = torch.device(config["network"]["device"])
    else:
      device = torch.device("cpu")
    use_alt_loss = config["network"]["use_alt_loss"]
    category_valid = utils.get_valid_categories()
    dict_cat2int, dict_int2cat, dict_cat2fun = utils.get_category_dicts(res)
    minvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_minvalues" + suffix_name + ".pkl", "rb" ))
    maxvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_maxvalues" + suffix_name + ".pkl", "rb" ))
    
    seq_starts = []
    seq_indices = []
    val_sequences = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_val_quantized" + suffix_name + ".pkl", "rb" ))
    room_masks_val = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_val_masks" + suffix_name + ".pkl", "rb" ))
    dataset_val = CustomDatasetPermute(val_sequences[:], max_seq_len,res,ergo_weights=[],room_masks=room_masks_val)
    val_sequences = torch.as_tensor(val_sequences)[:,:-1]
    rand_ind = torch.randperm(val_sequences.size(0))
    for j in range(n_sequences):
      seq = val_sequences[rand_ind[j],:].view(-1,6)
      keep_furn = seq[:,0] < 3 # room, window or door
      seq_starts.append(seq[keep_furn,:].flatten())
      seq_indices.append(rand_ind[j])
    room_mode = "replace_all_after"
    room_enc_dim = 1
    if "all" in room_mode:
      room_enc_dim = 5
    room_encodings = []
    posmap = torch.stack([torch.linspace(0.0,1.0,256).view(256,1).expand(-1,256),
                          torch.linspace(0.0,1.0,256).view(1,256).expand(256,-1)],dim=0)
    
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

    for model_n in model_names:
      model_name = "model_" + model_n
      if epoch == "best":
        val_loss_hist = pickle.load(open(path_trained_models + model_name + "/model_tmp" + str(n_versions-1) + "/val_loss_hist.pkl", "rb" ))
        n_epochs_skip = int(len(val_loss_hist) / 10)
        val_loss_hist_saved = val_loss_hist[n_epochs_skip-1::n_epochs_skip]
        epoch = torch.min(torch.as_tensor(val_loss_hist_saved),dim=0)[1].item()

      version_name = "model_tmp" + str(epoch)
      model = GPT2LMHeadModelCustom.from_pretrained(path_trained_models + model_name + "/" + version_name + "/")
      model.to(device)
      room_encoder = RoomEncoder(freeze_bn=False, input_channels=3, feature_size=64, res=room_enc_dim*256)
      room_encoder.load_state_dict(torch.load(path_trained_models + model_name + "/" + version_name + "/room_encoder.pt",map_location=device))
      room_encoder.to(device)
      
      sequences = torch.empty(0,max_seq_len,dtype=torch.long,device=model.device)
      seq_added = 0
      with torch.no_grad():
        model.eval()
        room_encoder.eval()
        while seq_added < n_sequences:
          print("Creating new scenes with model", model_n, "-", str(seq_added+1), "of" , str(n_sequences) , end="\r")
          nodes = None  
          scene_dict = None
          seq_start = seq_starts[seq_added].view(-1,6)
          if order_switch > 0:
            if order_switch == 1:
              seq_start = seq_start[:,[0,5,3,4,1,2]]
            if order_switch == 2:
              seq_start = seq_start[:,[0,5,1,2,3,4]]
          n_furn_start = seq_start.size(0)
          seq_start = seq_start.view(1,-1)
          nodes, collision, scene_dict = add_furniture(seq_start, nodes, scene_dict, model_source,add_all=True,collision_check=False, order_switch=order_switch)[:3]
          sequence = seq_start.to(model.device)
          pos_ids = torch.linspace(0,sequence.size(1)-1,sequence.size(1),dtype=torch.long,device=model.device).view(1,-1)
          ind_ids = pos_ids.clone() % 6
          cur_token = sequence.clone()
          past = None
          prev_past = None
          i = 0
          cont_count = 0
          post_windoor = True
          posmap = posmap.to(device)
          cur_masks = room_masks_val[seq_added].view(1,1,256,256).float().to(model.device)
          cur_masks = torch.cat([cur_masks,posmap.view(2,256,256).repeat([cur_masks.size(0),1,1,1])],dim=1)
          room_encoding = room_encoder(cur_masks)
          if "all" in room_mode:
            room_encoding=room_encoding.view(-1,room_enc_dim,res)
          room_encodings.append(room_encoding.view(-1,room_enc_dim,res)[0,:,:].detach().cpu())
          for furn_j in range(max_furniture+1-n_furn_start):
            max_noncat = True
            coll_count = 0
            furn_i = furn_j + n_furn_start
            collision = True
            while collision:
              while i < 6:
                input_ids = cur_token.clone()
                if input_ids.size(1) > 1:
                  output = model(input_ids,position_ids=pos_ids,index_ids=ind_ids,past_key_values=past,room_encodings=room_encoding,room_mode=room_mode)
                else:
                  output = model(input_ids,position_ids=pos_ids,index_ids=ind_ids,past_key_values=past)
                next_token_logits = output.logits[:,-1, :]
                filtered_next_token_logits = transformers.top_k_top_p_filtering(next_token_logits,top_k=top_k,top_p=top_p)
                probs = F.softmax(filtered_next_token_logits, dim=-1)
                cur_sample = torch.multinomial(probs, num_samples=1)
                if i == 0 and cur_sample < res and (cur_sample >= len(category_valid) or cur_sample <= 2):
                  cont_count = cont_count + 1
                  if cont_count > 20:
                    coll_count = cont_count
                    break
                  continue
                if i == 0 and cur_sample > 2:
                    post_windoor = True
                cur_token = cur_sample
                if max_noncat and (furn_i > 0) and (i > 0) and post_windoor:
                  cur_token_max = output.logits.argmax(axis=-1)
                  cur_token = cur_token_max
                past = output.past_key_values
                sequence = torch.cat([sequence,cur_token.clone()],1)
                if pos_ids.size(1) > 1:
                  pos_ids = pos_ids[:,[-1]]
                  ind_ids = ind_ids[:,[-1]]
                pos_ids = pos_ids + 1
                if cur_token == res:
                  ind_ids = torch.ones(1,1,dtype=torch.long,device=model.device) * (max_seq_len-1)
                else:
                  ind_ids = (ind_ids+1) % 6
                i = i + 1

              if cont_count > 20:
                break
              if cur_token < res:
                nodes, collision, scene_dict = add_furniture(sequence.clone().to('cpu'), nodes, scene_dict, model_source,collision_check=collision_check, order_switch=order_switch)[:3]
                if collision:
                  if furn_i > 0:
                    coll_count = coll_count + 1
                    sequence = sequence[:,:-6].clone().view(1,-1)
                    cur_token = sequence[:,[-1]].clone()
                    pos_ids = pos_ids - 6
                    past = prev_past
                  else:
                    coll_count = coll_count + 1
                    sequence = sequence[:,:-5].clone().view(1,-1)
                    cur_token = sequence[:,[-1]].clone()
                    pos_ids = pos_ids - 5
                    past = prev_past
                else:
                  prev_past = past
              else:
                collision = False
              if collision and furn_i == 0:
                i = 1
              else:
                i = 0
              if coll_count > 20:
                print("cannot be placed")  
                cont_count = cont_count + 1
                break
            if coll_count > 20:
              break
          
          has_failed = coll_count > 20 or cont_count > 20
          if not has_failed:
            sequence = torch.cat([sequence,res * torch.ones(1,max_seq_len-sequence.size(1),dtype=torch.long,device=model.device)],dim=1)
            sequences = torch.cat([sequences,sequence],dim=0)
            seq_added = seq_added + 1

      if order_switch > 0:
        last_tokens = sequences[:,-1]
        sequences = sequences[:,:-1].view(sequences.size(0),-1,6)
        if order_switch == 1:
          sequences = sequences[:,:,[0,4,5,2,3,1]].view(sequences.size(0),-1)
        else:
          sequences = sequences[:,:,[0,2,3,4,5,1]].view(sequences.size(0),-1)
        sequences = torch.cat([sequences,last_tokens.view(-1,1)],1)
      sequences = sequences.to('cpu')

      version_mean_scores = []
      version_median_scores = []
      version_variance_scores = []
      if order_switch > 0:
        last_tokens = sequences[:,-1]
        sequences = sequences[:,:-1].view(sequences.size(0),-1,6)
        if order_switch == 1:
          sequences = sequences[:,:,[0,4,5,2,3,1]].view(sequences.size(0),-1)
        else:
          sequences = sequences[:,:,[0,2,3,4,5,1]].view(sequences.size(0),-1)
        sequences = torch.cat([sequences,last_tokens.view(-1,1)],1)

      sequences = sequences.to('cpu')
      ergo_score = evaluate_scenes(sequences, minvalue_dict, maxvalue_dict, dict_cat2fun, use_log=True,grid_quantization=False, device=sequences.device,
        use_alt_loss=use_alt_loss)
      ergo_scores = torch.cat(ergo_score,0)
      scores_sorted, indices_sorted = torch.sort(ergo_scores)

      if not os.path.isdir(path_output_data):
        os.mkdir(path_output_data)
      if not os.path.isdir(path_output_data + model_name):
        os.mkdir(path_output_data + model_name)
      pickle.dump(scores_sorted.to('cpu'), open(path_output_data + model_name + "/resampled_" + sampling_type + "_scores.pkl", "wb" ))
      pickle.dump(sequences[indices_sorted,:].to('cpu'), open(path_output_data + model_name + "/resampled_" + sampling_type + "_sequences.pkl", "wb" ))
      pickle.dump(seq_indices, open(path_output_data + model_name + "/resampled_" + sampling_type + "_seq_indices.pkl", "wb" ))

      if args.num_room_plots > 0:
        if not os.path.isdir(config["paths"]["path_output_data"] + "/plots/"):
          os.mkdir(config["paths"]["path_output_data"] + "/plots/")
        if not os.path.isdir(config["paths"]["path_output_data"] + "/plots/" + model_name):
          os.mkdir(config["paths"]["path_output_data"] + "/plots/" + model_name)
        path_save_plots = config["paths"]["path_output_data"] + "/plots/" + model_name + "/" + house_name
        plot_sequences(sequences[:args.num_room_plots,:], dict_int2cat, dict_cat2int, minvalue_dict, maxvalue_dict, res, path_save_plots=path_save_plots)
      
if __name__ == "__main__":
  main(sys.argv[1:])