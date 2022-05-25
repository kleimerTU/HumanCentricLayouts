import argparse
import numpy as np
import os
import pickle
import sys
import torch
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from src import utils
from src.main_functions import *
from src.network import CustomDataset

def main(argv):
  parser = argparse.ArgumentParser(
      description="Convert rooms into sequences"
  )

  parser.add_argument(
      "path_to_config",
      nargs='+',
      type=str,
      help="Paths to the configuration files (accepts multiple inputs)"
  )
  
  parser.add_argument(
      "path_3dfront_maps",
      help="Path to the 3DFront room masks (see Readme for details)"
  )
  
  parser.add_argument(
      "--num_room_plots",
      default=0,
      type=int,
      help="Optionally plots the specified number of rooms for testing purposes"
  )
  
  args = parser.parse_args(argv)
  config_paths = args.path_to_config
  n_sets = len(config_paths)
  for i in range(n_sets):
    with open(config_paths[i], "r") as f:
      config = yaml.load(f, Loader=Loader)
    
    path_3dfront_data = config["paths"]["path_3dfront_data"] + "/"
    path_input_data = config["paths"]["path_input_data"] + "/"
    path_3dfront_maps = args.path_3dfront_maps + "/"
    set_name = config["general"]["set_name"]
    suffix_name = config["general"]["suffix_name"]
    max_furniture = config["general"]["max_furniture"]
    res = config["general"]["res"]
    max_seq_len = (max_furniture + 1) * 6 + 1
    category_valid = utils.get_valid_categories()
    dict_cat2int, dict_int2cat, dict_cat2fun = utils.get_category_dicts(res)
    
    # get furniture sequences, filter out rooms with too many objects or non-aligned windows or doors 
    print("Loading rooms")
    keep_rooms_with_inside_doors = True # if true, discards doors inside rooms instead of discarding the entire room
    train_sequences = []
    val_sequences = []
    minvalues = torch.empty(0,6)
    maxvalues = torch.empty(0,6)
    n_train = 0
    n_val = 0
    train_furniture = []
    val_furniture = []
    train_masks = []
    val_masks = []
    train_square = []
    val_square = []
    train_floor = []
    val_floor = []
    if set_name != "3dfront_all":
      furniture = pickle.load(open( path_3dfront_data + set_name + ".pkl", "rb" ))
      scene_info = pickle.load(open( path_3dfront_data + set_name + "_info.pkl", "rb" ))
      for i in range(len(furniture)):
        house_name = scene_info["house_name"][i].split(".")[0]
        folder_name = path_3dfront_maps + house_name + "_" + scene_info["room_name"][i]
        if os.path.isdir(folder_name):
          room_mask_img = Image.open(folder_name + "/room_mask.png").convert("RGB")
          room_mask =  np.asarray(room_mask_img).astype(np.float32) / np.float32(255)
          room_mask = room_mask[:, :, 0:1]
          room_mask = np.transpose(room_mask, (2, 0, 1)).astype(bool)
        else:
          continue
        scene_furniture = filter_furniture(furniture[i],category_valid,centering=True)
        for furn in scene_furniture:
          furn.pos = furn.pos.float()
          furn.dim = abs(furn.dim.float())
          furn.rot = furn.rot.float()
        if scene_info["is_train"][i]:
          train_furniture.append(scene_furniture)
          train_masks.append(room_mask)
          train_square.append(scene_info["is_square"][i])
          train_floor.append(scene_info["floor_mesh"][i])
        elif scene_info["is_val"][i]:
          if set_name == '3dfront_bedrooms':
            val_furniture.append(scene_furniture)
            val_masks.append(room_mask)
            val_square.append(scene_info["is_square"][i])
            val_floor.append(scene_info["floor_mesh"][i])
          else:
            train_furniture.append(scene_furniture)
            train_masks.append(room_mask)
            train_square.append(scene_info["is_square"][i])
            train_floor.append(scene_info["floor_mesh"][i])
        else:
          val_furniture.append(scene_furniture)
          val_masks.append(room_mask)
          val_square.append(scene_info["is_square"][i])
          val_floor.append(scene_info["floor_mesh"][i])
      furniture = []
    else: 
      raise NotImplementedError('Room masks are currently only supported for individual room types')

    valid_masks = []
    valid_square = []
    valid_floor = []
    for i in range(len(train_furniture)):
      print("Processing training set -", i+1, "of", len(train_furniture), "rooms processed",end="\r")
      furniture = train_furniture[i]
      if len(furniture) > 0 and len(furniture) <= max_furniture:
        all_aligned = True
        for furn in furniture:
          while furn.rot > np.pi:
            furn.rot = furn.rot - 2*np.pi
          while furn.rot <= -np.pi:
            furn.rot = furn.rot + 2*np.pi
          if furn.coarse_category == "window" or furn.coarse_category == "door":
            rot = furn.rot % (0.5*np.pi)
            if (rot > 0.025*np.pi) and (rot < 0.475*np.pi): # check if axis aligned
              all_aligned = False
        if all_aligned:
          room_seq = create_sequence(furniture,dict_cat2int,detailed=True,keep_room_dims=True,res=res)
          room_obj = sequence_to_furniture(room_seq[:6], dict_int2cat)[0]
          if room_obj.dim[0] > 20.0 or room_obj.dim[1] > 20.0: # skip rooms that are too large
            continue
          room_bb = room_obj.bbox()
          doors_valid = True
          valid_furniture = []
          for furn in furniture:
            if furn.coarse_category == "window" or furn.coarse_category == "door":
              door_bb = furn.bbox()
              is_inside = torch.logical_and(door_bb[0,:] > (room_bb[0,:] + 0.05), door_bb[1,:] < (room_bb[1,:] - 0.05))
              is_inside = torch.sum(is_inside,0) > 1
              valid_furniture.append(furn)
            else:
              valid_furniture.append(furn)
          if keep_rooms_with_inside_doors:
            doors_valid = True
          if doors_valid: # check if no door is completely inside the room (non-rectangular room)
            furniture = valid_furniture
            #furniture = adjust_windoors(room_bb, valid_furniture)
            room_seq = create_sequence(furniture,dict_cat2int,detailed=True,keep_room_dims=True,res=res)
            if torch.sum(torch.isnan(room_seq)) > 0:
              continue
            room_pos = room_seq[0,[1,2]]
            room_seq[:,[1,2]] = room_seq[:,[1,2]] - room_pos
            room_seq[0,5] = 0
            train_sequences.append(room_seq)
            mask = torch.as_tensor(train_masks[i].copy())
            valid_masks.append(mask)
            valid_square.append(train_square[i])
            valid_floor.append(train_floor[i])
            # augment dataset
            room_mask = train_masks[i]
            for r in range(4):
              room_mask = np.transpose(room_mask,(0,2,1))[:,:,::-1] # rotate
              for furn in furniture:
                furn.pos = torch.as_tensor([-furn.pos[1],furn.pos[0]])
                furn.rot = furn.rot + 0.5*np.pi
                if furn.rot > np.pi:
                  furn.rot = furn.rot - 2*np.pi
              if r < 3: # fourth rotation was already added at beginning
                furniture[0].dim = furniture[0].dim[[1,0]]
                furniture[0].rot = torch.zeros(1,)
                room_seq = create_sequence(furniture,dict_cat2int,detailed=True,keep_room_dims=True,res=res)
                room_pos = room_seq[0,[1,2]]
                room_seq[:,[1,2]] = room_seq[:,[1,2]] - room_pos
                room_seq[0,5] = 0
                train_sequences.append(room_seq)
                mask = torch.as_tensor(room_mask.copy())
                valid_masks.append(mask)
                valid_square.append(train_square[i])
                valid_floor.append(train_floor[i])
            # mirror and repeat
            for furn in furniture:
              furn.pos = torch.as_tensor([-furn.pos[0],furn.pos[1]])
              front_old = furn.world_front()
              front_new = torch.as_tensor([-front_old[0],front_old[1]])
              dot = torch.sum(front_old * front_new)
              det = front_old[0] * front_new[1] -  front_old[1] * front_new[0]
              furn.rot = furn.rot + torch.atan2(det, dot)
              if furn.rot > np.pi:
                furn.rot = furn.rot - 2*np.pi
            furniture[0].dim = furniture[0].dim[[1,0]]
            furniture[0].rot = torch.zeros(1,)
            room_mask = room_mask[:,:,::-1] # mirror
            for r in range(4):
              room_mask = np.transpose(room_mask,(0,2,1))[:,:,::-1] # rotate
              for furn in furniture:
                furn.pos = torch.as_tensor([-furn.pos[1],furn.pos[0]])
                furn.rot = furn.rot + 0.5*np.pi
                if furn.rot > np.pi:
                  furn.rot = furn.rot - 2*np.pi
              furniture[0].dim = furniture[0].dim[[1,0]]
              furniture[0].rot = torch.zeros(1,)
              room_seq = create_sequence(furniture,dict_cat2int,detailed=True,keep_room_dims=True,res=res)
              room_pos = room_seq[0,[1,2]]
              room_seq[:,[1,2]] = room_seq[:,[1,2]] - room_pos
              room_seq[0,5] = 0
              train_sequences.append(room_seq)
              mask = torch.as_tensor(room_mask.copy())
              valid_masks.append(mask)
              valid_square.append(train_square[i])
              valid_floor.append(train_floor[i])
      n_train = n_train + 1
    if not os.path.exists(path_input_data + set_name + "_data/"):
      os.mkdir(path_input_data + set_name + "_data/")
    pickle.dump(train_sequences, open(path_input_data + set_name + "_data/" + set_name + "_train_unquantized" + suffix_name + ".pkl", "wb"))
    pickle.dump(valid_masks, open(path_input_data + set_name + "_data/" + set_name + "_train_masks" + suffix_name + ".pkl", "wb"))
    pickle.dump(valid_square, open(path_input_data + set_name + "_data/" + set_name + "_train_square" + suffix_name + ".pkl", "wb"))
    pickle.dump(valid_floor, open(path_input_data + set_name + "_data/" + set_name + "_train_floor" + suffix_name + ".pkl", "wb"))
    print()
    
    valid_masks = []
    valid_square = []
    valid_floor = []
    for i in range(len(val_furniture)):
      print("Processing validation set -", i+1, "of", len(val_furniture), "rooms processed",end="\r")
      furniture = val_furniture[i]
      if len(furniture) > 0 and len(furniture) <= max_furniture:
        all_aligned = True
        for furn in furniture:
          while furn.rot > np.pi:
            furn.rot = furn.rot - 2*np.pi
          while furn.rot <= -np.pi:
            furn.rot = furn.rot + 2*np.pi
          if (furn.coarse_category == "window") or (furn.coarse_category == "door"):
            rot = furn.rot % (0.5*np.pi)
            if (rot > 0.025*np.pi) and (rot < 0.475*np.pi): # check if axis aligned
              all_aligned = False
        if all_aligned:
          room_seq = create_sequence(furniture,dict_cat2int,detailed=True,keep_room_dims=True,res=res)
          room_obj = sequence_to_furniture(room_seq[:6], dict_int2cat)[0]
          if room_obj.dim[0] > 20.0 or room_obj.dim[1] > 20.0: # skip rooms that are too large
            continue
          room_bb = room_obj.bbox()
          doors_valid = True
          valid_furniture = []
          for furn in furniture:
            if furn.coarse_category == "window" or furn.coarse_category == "door":
              door_bb = furn.bbox()
              is_inside = torch.logical_and(door_bb[0,:] > (room_bb[0,:] + 0.05), door_bb[1,:] < (room_bb[1,:] - 0.05))
              is_inside = torch.sum(is_inside,0) > 1
              valid_furniture.append(furn)
            else:
              valid_furniture.append(furn)
          if keep_rooms_with_inside_doors:
            doors_valid = True
          if doors_valid: # check if no door is completely inside the room (non-rectangular room)
            furniture = valid_furniture
            #furniture = adjust_windoors(room_bb, valid_furniture)
            room_seq = create_sequence(furniture,dict_cat2int,detailed=True,keep_room_dims=True,res=res)
            if torch.sum(torch.isnan(room_seq)) == 0:
              room_pos = room_seq[0,[1,2]]
              room_seq[:,[1,2]] = room_seq[:,[1,2]] - room_pos
              room_seq[0,5] = 0
              val_sequences.append(room_seq)
              mask = torch.as_tensor(val_masks[i].copy())
              valid_masks.append(mask)
              valid_square.append(val_square[i])
              valid_floor.append(val_floor[i])
              # augment dataset
              room_mask = val_masks[i]
              for r in range(4):
                room_mask = np.transpose(room_mask,(0,2,1))[:,:,::-1] # rotate
                for furn in furniture:
                  furn.pos = torch.as_tensor([-furn.pos[1],furn.pos[0]])
                  furn.rot = furn.rot + 0.5*np.pi
                  if furn.rot > np.pi:
                    furn.rot = furn.rot - 2*np.pi
                if r < 3: # fourth rotation was already added at beginning
                  furniture[0].dim = furniture[0].dim[[1,0]]
                  furniture[0].rot = torch.zeros(1,)
                  room_seq = create_sequence(furniture,dict_cat2int,detailed=True,keep_room_dims=True,res=res)
                  room_pos = room_seq[0,[1,2]]
                  room_seq[:,[1,2]] = room_seq[:,[1,2]] - room_pos
                  room_seq[0,5] = 0
                  val_sequences.append(room_seq)
                  mask = torch.as_tensor(room_mask.copy())
                  valid_masks.append(mask)
                  valid_square.append(val_square[i])
                  valid_floor.append(val_floor[i])
              # mirror and repeat
              for furn in furniture:
                furn.pos = torch.as_tensor([-furn.pos[0],furn.pos[1]])
                front_old = furn.world_front()
                front_new = torch.as_tensor([-front_old[0],front_old[1]])
                dot = torch.sum(front_old * front_new)
                det = front_old[0] * front_new[1] -  front_old[1] * front_new[0]
                furn.rot = furn.rot + torch.atan2(det, dot)
                if furn.rot > np.pi:
                  furn.rot = furn.rot - 2*np.pi
              furniture[0].dim = furniture[0].dim[[1,0]]
              furniture[0].rot = torch.zeros(1,)
              room_mask = room_mask[:,:,::-1] # mirror
              for r in range(4):
                room_mask = np.transpose(room_mask,(0,2,1))[:,:,::-1] # rotate
                for furn in furniture:
                  furn.pos = torch.as_tensor([-furn.pos[1],furn.pos[0]])
                  furn.rot = furn.rot + 0.5*np.pi
                  if furn.rot > np.pi:
                    furn.rot = furn.rot - 2*np.pi
                furniture[0].dim = furniture[0].dim[[1,0]]
                furniture[0].rot = torch.zeros(1,)
                room_seq = create_sequence(furniture,dict_cat2int,detailed=True,keep_room_dims=True,res=res)
                room_pos = room_seq[0,[1,2]]
                room_seq[:,[1,2]] = room_seq[:,[1,2]] - room_pos
                room_seq[0,5] = 0
                val_sequences.append(room_seq)
                mask = torch.as_tensor(room_mask.copy())
                valid_masks.append(mask)
                valid_square.append(val_square[i])
                valid_floor.append(val_floor[i])
      n_val = n_val + 1
    pickle.dump(val_sequences, open(path_input_data + set_name + "_data/" + set_name + "_val_unquantized" + suffix_name + ".pkl", "wb"))
    pickle.dump(valid_masks, open(path_input_data + set_name + "_data/" + set_name + "_val_masks" + suffix_name + ".pkl", "wb"))
    pickle.dump(valid_square, open(path_input_data + set_name + "_data/" + set_name + "_val_square" + suffix_name + ".pkl", "wb"))
    pickle.dump(valid_floor, open(path_input_data + set_name + "_data/" + set_name + "_val_floor" + suffix_name + ".pkl", "wb"))
    print()

    # for each category, get the range of possible values
    print("Quantizing sequences")
    train_sequences = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_train_unquantized" + suffix_name + ".pkl", "rb" ))
    val_sequences = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_val_unquantized" + suffix_name + ".pkl", "rb" ))
    categories = category_valid
    sequences_stacked = torch.cat([torch.cat(train_sequences,0),torch.cat(val_sequences,0)],0)
    minvalue_dict = dict()
    maxvalue_dict = dict()
    minpos = torch.empty(0,2)
    maxpos = torch.empty(0,2)
    # for i in range(len(categories)):
    #   indices = sequences_stacked[:,0] == i
    #   seq_of_cat = sequences_stacked[indices,:]
    #   if seq_of_cat.size(0) > 0:
    #     sorted = torch.sort(seq_of_cat,0)[0]
    #     minvalue_dict[i] = sorted[0,:]
    #     maxvalue_dict[i] = sorted[-1,:]
    #     #minvalue_dict[i] = sorted[int(seq_of_cat.size(0)*0.01),:]
    #     #maxvalue_dict[i] = sorted[-int(seq_of_cat.size(0)*0.01),:]
    #     #print(i,sorted[0,3:5],sorted[-1,3:5])
    #     minvalue_dict[i][torch.abs(minvalue_dict[i]) < 1e-05] = 0.0
    #     maxvalue_dict[i][torch.abs(maxvalue_dict[i]) < 1e-05] = 0.0
    #   else:
    #     minvalue_dict[i] = torch.zeros(6,)
    #     maxvalue_dict[i] = torch.ones(6,)
    # minvalues = torch.stack([minvalue_dict[i][1:6] for i in range(len(categories))],0)
    # maxvalues = torch.stack([maxvalue_dict[i][1:6] for i in range(len(categories))],0)
    # minpos = torch.min(minvalues,0)[0]
    # maxpos = torch.max(maxvalues,0)[0]
    sorted = torch.sort(sequences_stacked[:,1:6],0)[0]
    minpos = sorted[0,:]
    maxpos = sorted[-1,:]
    for i in range(len(categories)):
      minvalue_dict[i] = torch.cat([torch.as_tensor([i]),minpos.flatten()])
      maxvalue_dict[i] = torch.cat([torch.as_tensor([i]),maxpos.flatten()])
    pickle.dump(minvalue_dict, open(path_input_data + set_name + "_data/" + set_name + "_minvalues" + suffix_name + ".pkl", "wb"))
    pickle.dump(maxvalue_dict, open(path_input_data + set_name + "_data/" + set_name + "_maxvalues" + suffix_name + ".pkl", "wb"))

    # quantize and save
    minvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_minvalues" + suffix_name + ".pkl", "rb" ))
    maxvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_maxvalues" + suffix_name + ".pkl", "rb" ))
    quantized_train = []
    quantized_val = []
    for i in range(len(train_sequences)):
      if len(train_sequences[i]) > 0:
        i_sort = torch.argsort(train_sequences[i][:,0]) # sort by category
        quantized = quantize_sequence(train_sequences[i][i_sort,:],minvalue_dict,maxvalue_dict,res)
        quantized_train.append(quantized.flatten().tolist())
    for i in range(len(val_sequences)): 
      if len(val_sequences[i]) > 0: 
        i_sort = torch.argsort(val_sequences[i][:,0])
        quantized = quantize_sequence(val_sequences[i][i_sort,:],minvalue_dict,maxvalue_dict,res)
        quantized_val.append(quantized.flatten().tolist())
    pickle.dump(quantized_train, open(path_input_data + set_name + "_data/" + set_name + "_train_quantized" + suffix_name + ".pkl", "wb"))
    pickle.dump(quantized_val, open(path_input_data + set_name + "_data/" + set_name + "_val_quantized" + suffix_name + ".pkl", "wb"))

    # clean up if quantization went wrong somewhere
    cleanup = False
    if cleanup:
      train_sequences = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_train_quantized" + suffix_name + ".pkl", "rb" ))
      val_sequences = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_val_quantized" + suffix_name + ".pkl", "rb" ))
      minvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_minvalues" + suffix_name + ".pkl", "rb" ))
      maxvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_maxvalues" + suffix_name + ".pkl", "rb" ))
      minv = 1000
      maxv = -1000
      valid_train = []
      for j in range(len(train_sequences)):
        is_invalid = False
        for i in range(len(train_sequences[j])): 
          if train_sequences[j][i] < 0:
            is_invalid = True
        if not is_invalid:
          valid_train.append(train_sequences[j])
      valid_val = []
      for j in range(len(val_sequences)):
        is_invalid = False
        for i in range(len(val_sequences[j])):
          if val_sequences[j][i] < 0:
            print(j, val_sequences[j][i])
            is_invalid = True
        if not is_invalid:
          valid_val.append(val_sequences[j])
      pickle.dump(valid_train, open(path_input_data + set_name + "_data/" + set_name + "_train_quantized" + suffix_name + ".pkl", "wb"))
      pickle.dump(valid_val, open(path_input_data + set_name + "_data/" + set_name + "_val_quantized" + suffix_name + ".pkl", "wb"))
    
    # compute ergo scores
    print("Computing ergonomic scores")
    train_sequences = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_train_quantized" + suffix_name + ".pkl", "rb" ))
    val_sequences = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_val_quantized" + suffix_name + ".pkl", "rb" ))
    minvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_minvalues" + suffix_name + ".pkl", "rb" ))
    maxvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_maxvalues" + suffix_name + ".pkl", "rb" ))
    ergo_scores_new = []
    dataset = CustomDataset(train_sequences[:], max_seq_len,res)
    dataset_val = CustomDataset(val_sequences[:], max_seq_len,res)
    new_sequences_train = torch.as_tensor(train_sequences)
    new_sequences_val = torch.as_tensor(val_sequences)
    new_sequences_train[new_sequences_train > 1.0] = 1.0
    new_sequences_val[new_sequences_val > 1.0] = 1.0
    ergo_scores_new_train = evaluate_scenes(new_sequences_train, minvalue_dict, maxvalue_dict, dict_cat2fun, res=res, use_log=False, device='cpu')
    pickle.dump(ergo_scores_new_train, open(path_input_data + set_name + "_data/" + set_name + "_ergo_scores_train" + suffix_name + ".pkl", "wb"))
    ergo_scores_new_val = evaluate_scenes(new_sequences_val, minvalue_dict, maxvalue_dict, dict_cat2fun, res=res, use_log=False, device='cpu')
    pickle.dump(ergo_scores_new_val, open(path_input_data + set_name + "_data/" + set_name + "_ergo_scores_val" + suffix_name + ".pkl", "wb"))

  # plot some rooms as a test
  if args.num_room_plots > 0:
    train_sequences = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_train_quantized" + suffix_name + ".pkl", "rb" ))
    val_sequences = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_val_quantized" + suffix_name + ".pkl", "rb" ))
    minvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_minvalues" + suffix_name + ".pkl", "rb" ))
    maxvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_maxvalues" + suffix_name + ".pkl", "rb" ))
    for j in range(args.num_room_plots):
      i = 8 * j + 0
      quantized = torch.as_tensor(val_sequences[i]).view(-1,6)
      unquantized = unquantize_sequence(quantized,minvalue_dict,maxvalue_dict,res)
      test1 = sequence_to_furniture(unquantized,dict_int2cat)
      plot_room(test1,dict_cat2int)
    
if __name__ == "__main__":
    main(sys.argv[1:])