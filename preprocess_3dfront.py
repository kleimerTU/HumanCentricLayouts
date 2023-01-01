import argparse
import csv
import json
import numpy as np
import os
import pickle
import sys
import torch

from src.main_functions import *

categories_3d = [
    {'id': 1, 'super-category': 'Cabinet/Shelf/Desk', 'category': 'Children Cabinet', 'conv_cat': 'wardrobe_cabinet', 'ext_cat': 'children_cabinet'},
    {'id': 2, 'super-category': 'Cabinet/Shelf/Desk', 'category': 'Nightstand', 'conv_cat': 'stand', 'ext_cat': 'stand'},
    {'id': 3, 'super-category': 'Cabinet/Shelf/Desk', 'category': 'Bookcase / jewelry Armoire', 'conv_cat': 'shelving', 'ext_cat': 'shelving'},
    {'id': 4, 'super-category': 'Cabinet/Shelf/Desk', 'category': 'Wardrobe', 'conv_cat': 'wardrobe_cabinet', 'ext_cat': 'wardrobe_cabinet'},
    {'id': 5, 'super-category': 'Cabinet/Shelf/Desk', 'category': 'Coffee Table', 'conv_cat': 'table', 'ext_cat': 'coffee_table'},
    {'id': 6, 'super-category': 'Cabinet/Shelf/Desk', 'category': 'Corner/Side Table', 'conv_cat': 'table', 'ext_cat': 'side_table'},
    {'id': 7, 'super-category': 'Cabinet/Shelf/Desk', 'category': 'Sideboard / Side Cabinet / Console Table', 'conv_cat': 'dresser', 'ext_cat': 'sideboard'},
    {'id': 8, 'super-category': 'Cabinet/Shelf/Desk', 'category': 'Wine Cabinet', 'conv_cat': 'shelving', 'ext_cat': 'shelving'},
    {'id': 9, 'super-category': 'Cabinet/Shelf/Desk', 'category': 'TV Stand', 'conv_cat': 'tv_stand', 'ext_cat': 'tv_stand'},
    {'id': 10, 'super-category': 'Cabinet/Shelf/Desk', 'category': 'Drawer Chest / Corner cabinet', 'conv_cat': 'dresser', 'ext_cat': 'dresser'},
    {'id': 11, 'super-category': 'Cabinet/Shelf/Desk', 'category': 'Shelf', 'conv_cat': 'shelving', 'ext_cat': 'shelving'},
    {'id': 12, 'super-category': 'Cabinet/Shelf/Desk', 'category': 'Round End Table', 'conv_cat': 'ignore', 'ext_cat': 'side_table'},
    {'id': 13, 'super-category': 'Bed', 'category': 'King-size Bed', 'conv_cat': 'bed', 'ext_cat': 'bed'},
    {'id': 14, 'super-category': 'Bed', 'category': 'Bunk Bed', 'conv_cat': 'bed', 'ext_cat': 'kids_bed'},
    {'id': 15, 'super-category': 'Bed', 'category': 'Bed Frame', 'conv_cat': 'bed', 'ext_cat': 'bed'},
    {'id': 16, 'super-category': 'Bed', 'category': 'Single bed', 'conv_cat': 'bed', 'ext_cat': 'single_bed'},
    {'id': 17, 'super-category': 'Bed', 'category': 'Kids Bed', 'conv_cat': 'bed', 'ext_cat': 'kids_bed'},
    {'id': 18, 'super-category': 'Chair', 'category': 'Dining Chair', 'conv_cat': 'chair', 'ext_cat': 'chair'},
    {'id': 19, 'super-category': 'Chair', 'category': 'Lounge Chair / Cafe Chair / Office Chair', 'conv_cat': 'chair', 'ext_cat': 'lounge_chair'},
    {'id': 20, 'super-category': 'Chair', 'category': 'Dressing Chair', 'conv_cat': 'chair', 'ext_cat': 'special_chair'},
    {'id': 21, 'super-category': 'Chair', 'category': 'Classic Chinese Chair', 'conv_cat': 'chair', 'ext_cat': 'chair'},
    {'id': 22, 'super-category': 'Chair', 'category': 'Barstool', 'conv_cat': 'chair', 'ext_cat': 'barstool'},
    {'id': 23, 'super-category': 'Table', 'category': 'Dressing Table', 'conv_cat': 'dressing_table', 'ext_cat': 'dressing_table'},
    {'id': 24, 'super-category': 'Table', 'category': 'Dining Table', 'conv_cat': 'table', 'ext_cat': 'table'},
    {'id': 25, 'super-category': 'Table', 'category': 'Desk', 'conv_cat': 'desk', 'ext_cat': 'desk'},
    {'id': 26, 'super-category': 'Sofa', 'category': 'Three-Seat / Multi-seat Sofa', 'conv_cat': 'sofa', 'ext_cat': 'sofa'},
    {'id': 27, 'super-category': 'Sofa', 'category': 'armchair', 'conv_cat': 'sofa', 'ext_cat': 'armchair'},
    {'id': 28, 'super-category': 'Sofa', 'category': 'Loveseat Sofa', 'conv_cat': 'sofa', 'ext_cat': 'sofa'},
    {'id': 29, 'super-category': 'Sofa', 'category': 'L-shaped Sofa', 'conv_cat': 'sofa', 'ext_cat': 'shaped_sofa'},
    {'id': 30, 'super-category': 'Sofa', 'category': 'Lazy Sofa', 'conv_cat': 'sofa', 'ext_cat': 'special_chair'},
    {'id': 31, 'super-category': 'Sofa', 'category': 'Chaise Longue Sofa', 'conv_cat': 'sofa', 'ext_cat': 'sofa'},
    {'id': 32, 'super-category': 'Pier/Stool', 'category': 'Footstool / Sofastool / Bed End Stool / Stool', 'conv_cat': 'ottoman','ext_cat': 'ottoman'},
    {'id': 33, 'super-category': 'Lighting', 'category': 'Pendant Lamp', 'conv_cat': 'chandelier', 'ext_cat': 'chandelier'},
    {'id': 34, 'super-category': 'Lighting', 'category': 'Ceiling Lamp', 'conv_cat': 'chandelier', 'ext_cat': 'chandelier'},
    {'id': 35, 'super-category': 'Cabinet/Shelf/Desk', 'category': 'Shoe Cabinet', 'conv_cat': 'dresser', 'ext_cat': 'dresser'},
    {'id': 36, 'super-category': 'Bed', 'category': 'Couch Bed', 'conv_cat': 'sofa', 'ext_cat': 'sofa'},
    {'id': 37, 'super-category': 'Chair', 'category': 'Hanging Chair', 'conv_cat': 'ignore', 'ext_cat': 'special_chair'},
    {'id': 38, 'super-category': 'Chair', 'category': 'Folding chair', 'conv_cat': 'ignore', 'ext_cat': 'special_chair'},
    {'id': 39, 'super-category': 'Table', 'category': 'Bar', 'conv_cat': 'ignore', 'ext_cat': 'ignore'},
    {'id': 40, 'super-category': 'Sofa', 'category': 'U-shaped Sofa', 'conv_cat': 'sofa', 'ext_cat': 'shaped_sofa'},
    {'id': 41, 'super-category': 'Lighting', 'category': 'Floor Lamp', 'conv_cat': 'indoor_lamp', 'ext_cat': 'floor_lamp'},
    {'id': 42, 'super-category': 'Lighting', 'category': 'Wall Lamp', 'conv_cat': 'indoor_lamp', 'ext_cat': 'indoor_lamp'},
    {'id': 43, 'super-category': 'Chair', 'category': 'Lounge Chair / Book-chair / Computer Chair', 'conv_cat': 'chair', 'ext_cat': 'lounge_chair'},
    {'id': 44, 'super-category': 'Bed', 'category': 'Double Bed', 'conv_cat': 'bed', 'ext_cat': 'bed'},
    {'id': 45, 'super-category': 'Cabinet/Shelf/Desk', 'category': 'Sideboard / Side Cabinet / Console', 'conv_cat': 'dresser', 'ext_cat': 'sideboard'},
    {'id': 46, 'super-category': 'Sofa', 'category': 'Two-seat Sofa', 'conv_cat': 'sofa', 'ext_cat': 'sofa'},
    {'id': 47, 'super-category': 'Sofa', 'category': 'Three-Seat / Multi-person sofa', 'conv_cat': 'sofa', 'ext_cat': 'sofa'},
    {'id': 48, 'super-category': 'Table', 'category': 'Tea Table', 'conv_cat': 'table', 'ext_cat': 'coffee_table'},
    {'id': 49, 'super-category': 'Others', 'category': 'Wine Cooler', 'conv_cat': 'ignore', 'ext_cat': 'ignore'},
    {'id': 98, 'super-category': 'door', 'category': 'door', 'conv_cat': 'door', 'ext_cat': 'door'},
    {'id': 99, 'super-category': 'window', 'category': 'window', 'conv_cat': 'window', 'ext_cat': 'window'},
]

def main(argv):
    parser = argparse.ArgumentParser(
        description="Preprocess 3DFRONT data"
    )

    parser.add_argument(
        "path_3dfront_house",
        help="Path to 3DFRONT houses"
    )
    
    parser.add_argument(
        "path_3dfront_model",
        help="Path to 3DFRONT models"
    )
    
    parser.add_argument(
        "--path_3dfront_data",
        default="./data/3dfront",
        type=str,
        help="Where to save preprocessed 3DFRONT data"
    )
    
    parser.add_argument(
        "--room_type",
        default="bedrooms",
        type=str,
        help="The room type to process (bedrooms, livingrooms, diningrooms, libraries)"
    )
    
    parser.add_argument(
        "--create_dicts",
        action="store_true",
        help="Create the bbox dictionaries even if they exist already (this takes a while)"
    )
    
    args = parser.parse_args(argv)
    path_3dfront_house = args.path_3dfront_house + "/"
    path_3dfront_model = args.path_3dfront_model + "/"
    path_3dfront_data = args.path_3dfront_data + "/"
    
    room_type = args.room_type
    houses = dict(enumerate(os.listdir(path_3dfront_house)))
    with open(path_3dfront_model + "model_info.json") as file:
      model_info = json.load(file)
    if args.create_dicts or not os.path.isfile(path_3dfront_data + "model2bbox.json"): 
      create_dict_model2bbox(path_3dfront_model, path_3dfront_data)
    if args.create_dicts or not os.path.isfile(path_3dfront_data + "model2cat.json"): 
      create_dict_model2cat(model_info, houses, path_3dfront_house, path_3dfront_data)
    with open(path_3dfront_data + "model2cat.json") as file:
      dict_model2cat = json.load(file)
    with open(path_3dfront_data + "model2bbox.json") as file:
      model_bboxes = json.load(file)
    with open(path_3dfront_data + "invalid_rooms.txt") as file:
      invalid_rooms = file.readlines()
      invalid_rooms = [line.rstrip() for line in invalid_rooms]
    valid_rooms = []
    split_type = {}
    with open(path_3dfront_data + "valid_" + room_type + ".csv", "r") as f:
      data = csv.reader(f)
      for i in data:
        valid_rooms.append(i[0])
        split_type[i[0]] = i[1]

    converted_categories = []
    for i in range(len(categories_3d)):
      converted_categories.append(categories_3d[i]['ext_cat'])

    # dict that converts 3dfront categories to our categories
    dict_cat2cat = {}
    for i in range(len(categories_3d)):
      dict_cat2cat[categories_3d[i]["category"]] = converted_categories[i]

    # dicts that hold lists with model-id and dimensions for each category
    if args.create_dicts or not os.path.isfile(path_3dfront_data + "dict_cat2model_ext.pkl") or not os.path.isfile(path_3dfront_data + "dict_cat2dims_ext.pkl"): 
      create_dict_cat2model_cat2dims(model_bboxes, dict_cat2cat, dict_model2cat, path_3dfront_data)
      
    scene_info = {}
    scene_info["room_name"] = []
    scene_info["house_name"] = []
    scene_info["is_train"] = []
    scene_info["is_test"] = []
    scene_info["is_val"] = []
    scene_info["is_square"] = []
    scene_info["floor_mesh"] = []

    cat_count = {}
    device = "cpu"
    per_room_furniture = []
    for h in range(len(houses.keys())):
      print("Processing", room_type, "-", h+1, "of", str(len(houses.keys())), "files processed",end="\r")
      with open(path_3dfront_house + houses[h]) as file:
        data = json.load(file)
        uid2furn = {}
        for furn in data["furniture"]:
          uid2furn[furn["uid"]] = furn
          if furn["jid"] in dict_model2cat.keys():
            uid2furn[furn["uid"]]["category"] = dict_model2cat[furn["jid"]]
        id2mesh = {}
        door_positions = np.empty((0,3))
        door_dimensions = np.empty((0,3))
        window_positions = np.empty((0,3))
        window_dimensions = np.empty((0,3))
        for m in range(len(data['mesh'])):
          id2mesh[data['mesh'][m]['uid']] = data['mesh'][m]
          if data["mesh"][m]["type"] in ["Door","Window"]:
            xyz = np.asarray(data["mesh"][m]["xyz"])
            xyz = np.reshape(xyz,(-1,3))
            pos = np.mean(xyz,axis=0,keepdims=True)
            dim = np.max(xyz,axis=0,keepdims=True) - np.min(xyz,axis=0,keepdims=True)
            if data["mesh"][m]["type"] == "Window":
              window_positions = np.concatenate((window_positions,pos),axis=0)
              window_dimensions = np.concatenate((window_dimensions,dim),axis=0)
            if data["mesh"][m]["type"] == "Door":
              door_positions = np.concatenate((door_positions,pos),axis=0)
              door_dimensions = np.concatenate((door_dimensions,dim),axis=0)

      # gather all doors in the house
      all_doors = []
      for j in range(len(data["scene"]["room"])):
        room = data["scene"]["room"][j]
        for child in room["children"]:
          if child["instanceid"].startswith("furniture"):
            if child["ref"] in uid2furn.keys():
              if "category" in uid2furn[child["ref"]].keys() and uid2furn[child["ref"]]["category"] == 'door':
                if "valid" in uid2furn[child["ref"]].keys():
                  if uid2furn[child["ref"]]["valid"]:
                    all_doors.append(child)
                else:
                  all_doors.append(child)

      # process each room in the house
      for j in range(len(data["scene"]["room"])):
        augment_rolls = torch.rand((2,))
        furn_list = []
        converted_furniture = []
        room = data["scene"]["room"][j]
        furn_count = 0
        if room["instanceid"] in scene_info["room_name"]:
          continue
        if room["instanceid"] in invalid_rooms:
          continue
        if room["instanceid"] not in valid_rooms:
          continue
        # gather furniture
        for child in room["children"]:
          if child["instanceid"].startswith("furniture"):
            if child["ref"] in uid2furn.keys():
              child["jid"] = uid2furn[child["ref"]]["jid"]
              if "title" in uid2furn[child["ref"]].keys():
                child["title"] = uid2furn[child["ref"]]["title"]
              if "category" in uid2furn[child["ref"]].keys():
                child["category"] = uid2furn[child["ref"]]["category"]
              if "bbox" in uid2furn[child["ref"]].keys():
                child["bbox"] = uid2furn[child["ref"]]["bbox"]
              if "valid" in uid2furn[child["ref"]].keys():
                if uid2furn[child["ref"]]["valid"]:
                  furn_list.append(child)
              else:
                furn_list.append(child)

        # add room to the furniture list
        is_square = True
        v_room = np.empty((0,3))
        uv_room = np.empty((0,2))
        f_room = np.empty((0,3),dtype=int)
        for child in room["children"]:
          if child["instanceid"].startswith("mesh"):
            if "Floor" in id2mesh[child["ref"]]["type"]:
              xyz = np.asarray(id2mesh[child["ref"]]["xyz"],dtype=float)
              xyz = np.reshape(xyz,(-1,3))
              uv = np.asarray(id2mesh[child["ref"]]["uv"],dtype=float)
              uv = np.reshape(uv,(-1,2))
              faces = np.asarray(id2mesh[child["ref"]]["faces"],dtype=int)
              faces = np.reshape(faces,(-1,3))
              v_room = np.concatenate((v_room,xyz),axis=0)
              uv_room = np.concatenate((uv_room,uv),axis=0)
              f_room = np.concatenate((f_room,faces + f_room.shape[0]),axis=0)
        if v_room.shape[0] > 0:
          bbox_min = np.min(v_room,axis=0)
          bbox_max = np.max(v_room,axis=0)
          room_furn = Furniture()
          room_furn.coarse_category = "room"
          room_furn.fine_category = "room"
          room_furn.final_category = "room"
          room_furn.pos = torch.as_tensor(0.5*(bbox_min+bbox_max),dtype=torch.float32,device=device)[[0,2]]
          room_furn.dim = torch.as_tensor(bbox_max-bbox_min,dtype=torch.float32,device=device)[[0,2]]
          room_furn.rot = torch.as_tensor([0.0],dtype=torch.float32,device=device)
          room_furn.front = torch.as_tensor([0.0,1.0],dtype=torch.float32,device=device)
          converted_furniture.append(room_furn)
          # check if every corner has a floor vertex nearby to see if the room is square
          room_corners = room_furn.bbox()
          room_corners2 = room_corners.clone()
          room_corners2[:,1] = room_corners2[[1,0],1]
          room_corners = torch.concat([room_corners,room_corners2],dim=0)
          corner_dist = room_corners.view(1,-1,2) - torch.as_tensor(v_room[:,[0,2]]).view(-1,1,2)
          corner_dist = torch.linalg.norm(corner_dist,dim=2)
          corner_in_room = torch.sum(corner_dist < 0.5,dim=0) > 0
          is_square = torch.sum(corner_in_room) == 4

        # add furniture objects to the furniture list
        chandeliers = []
        lamps = []
        for furn in furn_list:
          if "category" not in furn.keys():
            continue
          if "title" in furn.keys() and furn["title"].startswith("window"):
            category = "window"
          elif "title" in furn.keys() and furn["title"].startswith("door"):
            category = "door"
          else:
            category = dict_model2cat[furn["jid"]]
          if category is None:
            continue
          if category == 'door':
            continue
          new_furn = Furniture()
          new_furn.fine_category = dict_cat2cat[category]
          new_furn.final_category = dict_cat2cat[category]
          new_furn.coarse_category = dict_cat2cat[category]
          new_furn.pos = torch.as_tensor(furn["pos"],device=device)[[0,2]]
          furn_scale = furn["scale"]
          if category == "window":
            furn_scale = [1.0,1.0,1.0]
            if window_positions.shape[0] > 0:
              dist_vec = new_furn.pos.view(1,2) - torch.as_tensor(window_positions)[:,[0,2]]
              dist = torch.linalg.norm(dist_vec,dim=-1)
              min_dist, min_index = torch.min(dist,axis=0)
              bb = torch.as_tensor(window_dimensions)[min_index,[0,2]]
              bb_sorted, _ = torch.sort(bb,descending=True) # order of bb dimensions is not always correct
              furn_bbox = [bb_sorted[0],1.25,bb_sorted[1]]
            else:
              furn_bbox = [0.75,1.25,0.2]
          else:
            furn_bbox = model_bboxes[furn["jid"]]
            if len(furn_bbox) == 1:
              furn_bbox = furn_bbox[0]
            if len(furn_scale) == 1:
              furn_scale = furn_scale[0]
          new_furn.dim = torch.as_tensor(furn_bbox,device=device)[[0,2]] * torch.as_tensor(furn_scale,device=device)[[0,2]]
          rot = furn["rot"]
          axis = torch.as_tensor(np.cross([0,0,1], rot[1:]),device=device,dtype=torch.float32)
          rot_sign = np.sign(axis[1])
          if rot_sign > -0.5:
            rot_sign = 1.0
          new_furn.rot = torch.as_tensor(-rot_sign * np.arccos(np.dot([0,0,1], rot[1:]))*2,device=device,dtype=torch.float32).view(1,)
          new_furn.front = torch.as_tensor([0.0,1.0],device=device)
          if category == "window":
            new_furn.rot = (new_furn.rot + np.pi) % (2*np.pi)
          if torch.sum(torch.isnan(new_furn.pos)) + torch.sum(torch.isnan(new_furn.dim)) + torch.sum(torch.isnan(new_furn.rot)) == 0:
            converted_furniture.append(new_furn)
            if category != "window":
              furn_count = furn_count + 1
            if category not in cat_count.keys():
              cat_count[category] = 0
            cat_count[category] += 1
            if new_furn.coarse_category == "chandelier":
              chandeliers.append(new_furn)

            # augment dataset with other objects
            if new_furn.coarse_category == "tv_stand":
              tv_furn = Furniture()
              tv_furn.fine_category = "television"
              tv_furn.final_category = "television"
              tv_furn.coarse_category = "television"
              tv_furn.pos = new_furn.pos.clone()
              tv_furn.dim = torch.as_tensor([1.0,0.1],device=device)
              tv_furn.rot = new_furn.rot.clone()
              tv_furn.front = new_furn.front.clone()
              converted_furniture.append(tv_furn)
              furn_count = furn_count + 1
              if category not in cat_count.keys():
                cat_count[category] = 0
              cat_count[category] += 1
            if new_furn.coarse_category == "desk" and augment_rolls[0] >= 0.5:
              tv_furn = Furniture()
              tv_furn.fine_category = "computer"
              tv_furn.final_category = "computer"
              tv_furn.coarse_category = "computer"
              tv_furn.pos = new_furn.pos.clone()
              tv_furn.dim = torch.as_tensor([0.5,0.3],device=device)
              tv_furn.rot = new_furn.rot.clone()
              tv_furn.front = new_furn.front.clone()
              converted_furniture.append(tv_furn)
              furn_count = furn_count + 1
              if category not in cat_count.keys():
                cat_count[category] = 0
              cat_count[category] += 1
            if new_furn.coarse_category in ["stand","side_table"] and augment_rolls[1] >= 0.5:
              tv_furn = Furniture()
              tv_furn.fine_category = "indoor_lamp"
              tv_furn.final_category = "indoor_lamp"
              tv_furn.coarse_category = "indoor_lamp"
              tv_furn.pos = new_furn.pos.clone()
              tv_furn.dim = torch.as_tensor([0.2,0.2],device=device)
              tv_furn.rot = new_furn.rot.clone()
              tv_furn.front = new_furn.front.clone()
              lamps.append(tv_furn)
        
        # only add lamps if there is not already a chandelier
        for lamp in lamps:
          intersecting = False
          for chandelier in chandeliers:
            intersect_area = get_intersection_area2d(lamp.bbox(), chandelier.bbox())
            if intersect_area > 0.01:
              intersecting = True
          if not intersecting:
            converted_furniture.append(lamp)
            furn_count = furn_count + 1
            if lamp.coarse_category not in cat_count.keys():
              cat_count[lamp.coarse_category] = 0
            cat_count[lamp.coarse_category] += 1

        # find out which doors belong to the room and add them
        for furn in all_doors:
          category = 'door'
          new_furn = Furniture()
          new_furn.fine_category = dict_cat2cat[category]
          new_furn.final_category = dict_cat2cat[category]
          new_furn.coarse_category = dict_cat2cat[category]
          new_furn.pos = torch.as_tensor(furn["pos"],device=device)[[0,2]]
          if door_positions.shape[0] > 0:
            dist_vec = new_furn.pos.view(1,2) - torch.as_tensor(door_positions)[:,[0,2]]
            dist = torch.linalg.norm(dist_vec,dim=-1)
            min_dist, min_index = torch.min(dist,axis=0)
            bb = torch.as_tensor(door_dimensions)[min_index,[0,2]]
            bb_sorted, _ = torch.sort(bb,descending=True) # order of bb dimensions is not always correct
            furn_bbox = [bb_sorted[0],2.0,bb_sorted[1]]
          else:
            furn_bbox = [0.75,2.0,0.2]
          furn_scale = [1.0,1.0,1.0]
          new_furn.dim = torch.as_tensor(furn_bbox,device=device)[[0,2]] * torch.as_tensor(furn_scale,device=device)[[0,2]]
          rot = furn["rot"]
          axis = torch.as_tensor(np.cross([0,0,1], rot[1:]),device=device,dtype=torch.float32)
          rot_sign = np.sign(axis[1])
          if rot_sign > -0.5:
            rot_sign = 1.0
          new_furn.rot = torch.as_tensor(-rot_sign * np.arccos(np.dot([0,0,1], rot[1:]))*2,device=device,dtype=torch.float32).view(1,)
          new_furn.front = torch.as_tensor([0.0,1.0],device=device)
          door_outside = False
          outside_thresh = 0.2
          is_horizontal = torch.abs(torch.dot(new_furn.front,new_furn.world_front())) > 0.9
          is_vertical = torch.abs(torch.dot(new_furn.front,new_furn.world_front())) < 0.1
          if is_horizontal:
            door_outside = door_outside or (new_furn.pos[0] < bbox_min[0])
            door_outside = door_outside or (new_furn.pos[0] > bbox_max[0])
            door_outside = door_outside or (new_furn.pos[1] < bbox_min[2] - outside_thresh)
            door_outside = door_outside or (new_furn.pos[1] > bbox_max[2] + outside_thresh)
          elif is_vertical:
            door_outside = door_outside or (new_furn.pos[0] < bbox_min[0] - outside_thresh)
            door_outside = door_outside or (new_furn.pos[0] > bbox_max[0] + outside_thresh)
            door_outside = door_outside or (new_furn.pos[1] < bbox_min[2])
            door_outside = door_outside or (new_furn.pos[1] > bbox_max[2])
          else:
            continue
          if not door_outside:
            converted_furniture.append(new_furn)
            if category not in cat_count.keys():
              cat_count[category] = 0
            cat_count[category] += 1
        if furn_count > 1:
          per_room_furniture.append(converted_furniture)
          scene_info["room_name"].append(room["instanceid"])
          scene_info["house_name"].append(houses[h])
          scene_info["is_train"].append(split_type[room["instanceid"]] == "train")
          scene_info["is_val"].append(split_type[room["instanceid"]] == "val")
          scene_info["is_test"].append(split_type[room["instanceid"]] == "test")
          scene_info["is_square"].append(is_square)
          scene_info["floor_mesh"].append({"v":v_room,"uv":uv_room,"f":f_room})
      
    pickle.dump(per_room_furniture, open(path_3dfront_data + "3dfront_" + room_type + ".pkl", "wb" ))
    pickle.dump(scene_info, open(path_3dfront_data + "3dfront_" + room_type + "_info.pkl", "wb" ))

    # plot some rooms as a test
    test_plot = False
    if test_plot:
      for i in range(20):
        print(scene_info["house_name"][i])
        print(scene_info["room_name"][i])
        print(scene_info["is_square"][i])
        plot_room(per_room_furniture[i], dict_cat2int)
    
def create_dict_model2cat(model_info, houses, houses_path, out_path):
  """
  Creates a dictionary that converts model-id to category and saves it as a json file
  """
  print("Creating model-to-category info")
  dict_model2cat = {}
  for model in model_info:
    if model["model_id"] == 'a360edba-7517-4d36-8063-6a0d89beeba0':
      dict_model2cat[model["model_id"]] = 'Dining Table'
    else:
      dict_model2cat[model["model_id"]] = model["category"]
  for h in range(len(houses.keys())):
    with open(houses_path + houses[h]) as file:
      data = json.load(file)
      for furn in data["furniture"]:
        if "jid" not in furn.keys():
          continue
        if "title" in furn.keys() and furn["title"].startswith("window"):
          dict_model2cat[furn["jid"]] = "window"
        if "title" in furn.keys() and furn["title"].startswith("door"):
          dict_model2cat[furn["jid"]] = "door"
    if (h % 1) == 0:
      print(h+1, "of", len(houses.keys()), "files processed",end="\r")
  print()
  with open(out_path + "model2cat.json","w") as file:
    json.dump(dict_model2cat, file, indent=4)

def create_dict_model2bbox(model_path, out_path):
  """
  Creates dictionaries that convert model-id to bbox info and saves them as json files
  """
  folders = dict(enumerate(os.listdir(model_path)))
  dict_model2bbox = {}
  dict_model2minp = {}
  dict_model2maxp = {}
  for i in range(len(folders.keys())):
    if (i % 1) == 0:
        print("Creating model bbox info -", i, "of", len(folders.keys()), "files processed",end="\r")
    if os.path.isdir(model_path + folders[i]):
        with open(model_path + folders[i] + '/raw_model.obj', 'r') as file:
            vertices = np.empty((0,3))
            lines = file.readlines()
            for line in lines:
                words = line.split()
                if len(words) < 1:
                    continue
                if words[0] == 'v':
                    v = np.reshape(np.asarray([float(words[1]),float(words[2]),float(words[3])]),(1,3))
                    vertices = np.concatenate((vertices,v),axis=0)
            bbox = np.max(vertices, axis=0) - np.min(vertices, axis=0)
            dict_model2bbox[folders[i]] = bbox.tolist()
            dict_model2minp[folders[i]] = np.min(vertices, axis=0).tolist()
            dict_model2maxp[folders[i]] = np.max(vertices, axis=0).tolist()
  dict_model2bbox["ts_computer"] = [0.6495,0.5217146300000001,0.359507]
  dict_model2bbox["ts_tv"] = [0.99,0.7038720022258601,0.0996884]
  dict_model2bbox["ts_door0"] =  [1.060501,2.11598,0.146625]
  dict_model2bbox["ts_door1"] =  [2.121602,2.11598,0.146625]
  dict_model2bbox["ts_window0"] =  [0.84702568,1.1968835549999999,0.22399480500000002]
  dict_model2bbox["ts_window1"] =  [0.4731994319999999,1.2620006400000001,0.15]
  dict_model2bbox["ts_window2"] =  [1.961768,0.901153,0.12717]
  
  dict_model2minp["ts_computer"] = [-0.32475,-0.00222363,-0.180023]
  dict_model2minp["ts_tv"] = [-0.495,-2.22586e-09,-0.0498509]
  dict_model2minp["ts_door0"] =  [-0.530401,0.0,-0.07335]
  dict_model2minp["ts_door1"] =  [-1.060801,0.0,-0.07335]
  dict_model2minp["ts_window0"] =  [-0.42351284,-1.2835000000000002e-05,-0.11302127]
  dict_model2minp["ts_window1"] =  [-0.23659971599999996,0.0,-0.075]
  dict_model2minp["ts_window2"] =  [-0.980884,0.0,-0.063585]
  
  dict_model2maxp["ts_computer"] = [0.32475,0.519491,0.179484]
  dict_model2maxp["ts_tv"] = [0.495,0.703872,0.0498375]
  dict_model2maxp["ts_door0"] =  [0.530401,2.11598,0.073275]
  dict_model2maxp["ts_door1"] =  [1.060801,2.11598,0.073275]
  dict_model2maxp["ts_window0"] =  [0.42351284,1.19687072,0.11097353500000001]
  dict_model2maxp["ts_window1"] =  [0.23659971599999996,1.2620006400000001,0.075]
  dict_model2maxp["ts_window2"] =  [0.980884,0.901153,0.063585]
  print()
  with open(out_path + "model2bbox.json","w") as file:
      json.dump(dict_model2bbox, file, indent=4)
  with open(out_path + "model2minp.json","w") as file:
      json.dump(dict_model2minp, file, indent=4)
  with open(out_path + "model2maxp.json","w") as file:
      json.dump(dict_model2maxp, file, indent=4)


def create_dict_cat2model_cat2dims(model_bboxes, dict_cat2cat, dict_model2cat, out_path):
  """
  Creates dictionaries that hold lists with model-id and dimensions for each category
  """
  dict_cat2model = {}
  dict_cat2dims = {}
  for key in model_bboxes.keys():
    if key in dict_model2cat.keys() and dict_model2cat[key] and not dict_model2cat[key] == "Bed Frame":
      category = dict_cat2cat[dict_model2cat[key]]
      bbox = model_bboxes[key]
      bbox = [bbox[0],bbox[2],bbox[1]]
      if category not in dict_cat2model.keys():
        dict_cat2model[category] = []
        dict_cat2dims[category] = []
      dict_cat2model[category].append(key)
      dict_cat2dims[category].append(torch.as_tensor(bbox,dtype=torch.float32))
  dict_cat2model["window"] = ["ts_window0","ts_window1","ts_window2"]
  dict_cat2model["door"] = ["ts_door0","ts_door1"]
  dict_cat2model["television"] = ["ts_tv"]
  dict_cat2model["computer"] = ["ts_computer"]
  dict_cat2dims["window"] = [torch.as_tensor([0.85,1.20,0.22]),torch.as_tensor([0.47,1.26,0.15]),torch.as_tensor([1.86,0.9,0.13])]
  dict_cat2dims["door"] = [torch.as_tensor([1.06,2.12,0.15]),torch.as_tensor([2.12,2.12,0.15])]
  dict_cat2dims["television"] = [torch.as_tensor([1.0,0.5,0.1])]
  dict_cat2dims["computer"] = [torch.as_tensor([0.5,0.4,0.3])]
  dict_cat2model["indoor_lamp"] = []
  dict_cat2dims["indoor_lamp"] = []
  for i in range(len(dict_cat2model["floor_lamp"])):
    lamp_model = dict_cat2model["floor_lamp"][i]
    lamp_dims = dict_cat2dims["floor_lamp"][i]
    if lamp_dims[2] < 0.8:
      dict_cat2model["indoor_lamp"].append(lamp_model)
      dict_cat2dims["indoor_lamp"].append(lamp_dims)
  pickle.dump(dict_cat2model, open(out_path + "dict_cat2model_ext.pkl", "wb"))
  pickle.dump(dict_cat2dims, open(out_path + "dict_cat2dims_ext.pkl", "wb"))

if __name__ == "__main__":
    main(sys.argv[1:])