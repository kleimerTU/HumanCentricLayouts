import json
import torch

from src.main_functions import *

def add_furniture(sequence, minvalue_dict, maxvalue_dict, dict_int2cat, dict_cat2model, dict_cat2dims, model2minp, model2maxp, res=256, nodes=None, scene_dict=None, invalid_models=[], add_all=False, collision_check=True, order_switch=1):
  """
  Adds a furniture object into the scene, optionally checking for collisions
  """
  orthographic_mode = False
  room_seq = sequence[[-1],:].clone().view(-1,6)
  if order_switch > 0:
    if order_switch == 1:
      room_seq = room_seq[:,[0,4,5,2,3,1]]
    else:
      room_seq = room_seq[:,[0,2,3,4,5,1]]
  valid_row = torch.logical_and(room_seq[:,0] < len(minvalue_dict.keys()),torch.sum(room_seq >= res,1) < 1)
  if torch.sum(valid_row) < 1:
    return nodes, True, scene_dict, None
  room_seq = room_seq[valid_row,:]
  reconstructed = unquantize_sequence_grid(room_seq,minvalue_dict,maxvalue_dict,res)
  furniture = sequence_to_furniture(reconstructed,dict_int2cat)
  if scene_dict == None or len(scene_dict.keys()) == 0:
    scene_dict = {}
    scene_dict["bb_in_scene"] = []
    scene_dict["bb_is_supporter"] = []
    scene_dict["bb_is_supported"] = []
    scene_dict["bb_is_near_ceil"] = []
    scene_dict["bb_is_chair"] = []
    scene_dict["bb_is_table"] = []
    scene_dict["bb_is_sofa"] = []
    scene_dict["bb_is_window"] = []
    scene_dict["furn_added"] = []
    scene_dict["furn_heights"] = []
    scene_dict["furn_transforms"] = []

  cat_supporter = ['table','desk','coffee_table','side_table','dresser','dressing_table','sideboard','shelving','stand','tv_stand']
  cat_supported = ['television','computer','indoor_lamp']
  cat_near_ceil = ['chandelier']
  cat_table = ['table','desk','dressing_table','coffee_table','side_table']
  cat_chair = ['chair','lounge_chair','special_chair','armchair','ottoman','shaped_sofa']
  cat_sofa = ['shaped_sofa']

  if nodes == None:
    nodes = []
  room_height = torch.as_tensor([2.5])
  wall_thick = 0.05
  if len(invalid_models) == 0:
    with open(path_3dfront_data + "invalid_models.txt") as file:
      invalid_model_lines = file.readlines()
      invalid_models.extend([line.rstrip() for line in invalid_model_lines])
  collision = False
  if add_all:
    to_add = range(len(furniture))
  else:
    to_add = [len(furniture)-1]
  for f in to_add:
    furn = furniture[f]
    if furn.coarse_category != 'room':
      models = dict_cat2model[furn.coarse_category]
      is_valid = [(model not in invalid_models) for model in models]
      models = [models[m] for m in range(len(models)) if is_valid[m]]
      model_dims = torch.stack(dict_cat2dims[furn.coarse_category],0)
      model_dims = model_dims[is_valid,:]
      if (furn.coarse_category != 'door') and (furn.coarse_category != 'window'):
        furn_ratio = furn.dim[0] / furn.dim[1]
        furn_area = furn.dim[0] * furn.dim[1]
        model_ratios = model_dims[:,0] / model_dims[:,1]
        model_areas = model_dims[:,0] * model_dims[:,1]
        model_errors = (model_dims[:,0]-furn.dim[0])**2 + (model_dims[:,1]-furn.dim[1])**2 + 0.01 * (model_ratios[:]-furn_ratio)**2
      else:
        model_ratios = model_dims[:,0]
        model_areas = model_dims[:,0]
        model_errors = (model_dims[:,0]-furn.dim[0])**2
        furn_ratio = furn.dim[0]
        furn_area = furn.dim[0]
      error_sorted, error_indices_sorted = torch.sort(model_errors)

      for best_model_index in range(np.minimum(len(error_sorted),1)):
        best_model = models[error_indices_sorted[best_model_index]]
        best_model_dim = model_dims[error_indices_sorted[best_model_index]]

        best_model_front = np.asarray([0,0,1.0])
        best_model_front = torch.as_tensor(best_model_front,dtype=torch.float32)
        furn_dir = furn.world_front()

        best_model_front = torch.as_tensor([best_model_front[0],best_model_front[2]])
        rotdir = torch.sign(best_model_front[0] * furn_dir[1] - best_model_front[1] * furn_dir[0])
        if rotdir > -1:
          rotdir = 1
        rot = torch.as_tensor([rotdir]) * torch.acos(torch.dot(best_model_front,furn_dir))

        model_min_p = model2minp[best_model]
        model_max_p = model2maxp[best_model]
        model_min_p = torch.as_tensor(model_min_p,dtype=torch.float32)
        model_max_p = torch.as_tensor(model_max_p,dtype=torch.float32)
        model_height = model_max_p[1] - model_min_p[1]
        scale_y = 1.0
        if model_height > room_height:
          scale_y = room_height / model_height
          model_height = room_height
        furn_pos_y = torch.zeros(1,)
        if furn.coarse_category == 'chandelier':
          furn_pos_y = torch.maximum(room_height - model_height,torch.as_tensor([2.0]))
        if furn.coarse_category == 'shelving' and model_height < 0.5:
          furn_pos_y = 1.60
        if furn.coarse_category == 'window':
          if model_height > (1.0*room_height/2.0):
            scale_y = (1.0*room_height/2.0) / model_height
            model_height = (1.0*room_height/2.0)
          furn_pos_y = room_height - model_height - (room_height * 0.1)
          if not orthographic_mode:
            pos_vec = (furn.pos - 0.5*furniture[0].dim) / torch.linalg.vector_norm((furn.pos - 0.5*furniture[0].dim))
            t_sign = -torch.sign(torch.dot(pos_vec,furn_dir))
            if t_sign > -1.0:
              t_sign = 1.0
            furn.pos = furn.pos + furn_dir * t_sign * 1.0 * furn.dim[1]

        if best_model_index > 0: # don't scale if best fit collides with scene
          scale_w = 1.0
          scale_h = 1.0
        else:
          scale_w = furn.dim[0] / best_model_dim[0]
          scale_h = furn.dim[1] / best_model_dim[1]
          if furn.coarse_category == 'door':
            scale_h = 1.0
        if (furn.coarse_category == 'window') and not orthographic_mode:
          scale_h = 0.3

        if furn.coarse_category in ['desk','shaped_sofa','wardrobe_cabinet']:
          pos_vec = (furn.pos - 0.5*furniture[0].dim) / torch.linalg.vector_norm((furn.pos - 0.5*furniture[0].dim))
          furn_front = furn.world_front()
          scale_sign = torch.sign(torch.dot(pos_vec,furn_front))
          if scale_sign > -1.0:
            scale_sign = 1.0
          scale_w = scale_sign *  scale_w
        
        S = torch.as_tensor([[scale_w,0.0,0.0,0.0], [0.0,scale_y,0.0,0.0], [0.0,0.0,scale_h,0.0], [0.0,0.0,0.0,1.0]])
        T = torch.as_tensor([[1.0,0.0,0.0,furn.pos[0]], [0.0,1.0,0.0,furn_pos_y], [0.0,0.0,1.0,furn.pos[1]], [0.0,0.0,0.0,1.0]])
        R = torch.as_tensor([[torch.cos(rot),0.0,-torch.sin(rot),0.0], [0.0,1.0,0.0,0.0], [torch.sin(rot),0.0,torch.cos(rot),0.0], [0.0,0.0,0.0,1.0]])
        TRS = torch.matmul(T,torch.matmul(R,S))

        min_p = torch.as_tensor([-best_model_dim[0]/2.0,0.0,-best_model_dim[1]/2.0,1.0])
        max_p = torch.as_tensor([best_model_dim[0]/2.0,0.0,best_model_dim[1]/2.0,1.0])
        min_p2 = torch.as_tensor([best_model_dim[0]/2.0,0.0,-best_model_dim[1]/2.0,1.0])
        max_p2 = torch.as_tensor([-best_model_dim[0]/2.0,0.0,best_model_dim[1]/2.0,1.0])
        min_p = torch.matmul(TRS,min_p)
        max_p = torch.matmul(TRS,max_p)
        min_p2 = torch.matmul(TRS,min_p2)
        max_p2 = torch.matmul(TRS,max_p2)
        model_corners = torch.stack([min_p[:3],max_p[:3],min_p2[:3],max_p2[:3]])
        model_bb = torch.stack([torch.min(model_corners,dim=0)[0],torch.max(model_corners,dim=0)[0]])

        furn_corners = furn.bbox()
        furn_corners = torch.cat([furn_corners[:,[0]],torch.as_tensor([0.0,1.0]).view(2,1),furn_corners[:,[1]]],1)
        furn_bbox = {}
        furn_bbox["min"] = furn_corners[0,:].tolist()
        furn_bbox["max"] = furn_corners[1,:].tolist()

        collision = False
        if furn.coarse_category == 'door' or furn.coarse_category == 'window':
          min_dist = torch.min(torch.abs(torch.as_tensor([furn.pos[0],furn.pos[1],furn.pos[0]-furniture[0].dim[0],furn.pos[1]-furniture[0].dim[1]])))
          if min_dist >= 2*furn.dim[1]: # check if door or window not close to wall
            collision = True
        else:
          room_corners = torch.stack([scene_dict["room_min"],scene_dict["room_max"]])
          model_area = (model_bb[1,0]-model_bb[0,0]) * (model_bb[1,2]-model_bb[0,2])
          inter_area = get_intersection_area(model_bb, room_corners)
          if collision_check and model_area * 0.95 > inter_area:
            collision = True
        if not collision:
          bb_in_scene = scene_dict["bb_in_scene"]
          for bb in range(len(bb_in_scene)):
            do_check = False
            if not furn.coarse_category == 'rug':
              do_check = True
            if furn.coarse_category != 'door' and furn.coarse_category != 'window' and scene_dict["bb_is_window"][bb]:
              do_check = False
            if (scene_dict["bb_is_supporter"][bb] and (furn.coarse_category in cat_supported)) or (scene_dict["bb_is_supported"][bb] and (furn.coarse_category in cat_supporter)):
              do_check = False
            if (scene_dict["bb_is_chair"][bb] and (furn.coarse_category in cat_table)) or (scene_dict["bb_is_table"][bb] and (furn.coarse_category in cat_chair)):
              do_check = False
            if (scene_dict["bb_is_near_ceil"][bb] and (furn.coarse_category not in cat_near_ceil)) or (not scene_dict["bb_is_near_ceil"][bb] and (furn.coarse_category in cat_near_ceil)):
              do_check = False
            if do_check:
              model_area = (model_bb[1,0]-model_bb[0,0]) * (model_bb[1,2]-model_bb[0,2])
              bb_in_scene_area = (bb_in_scene[bb][1,0]-bb_in_scene[bb][0,0]) * (bb_in_scene[bb][1,2]-bb_in_scene[bb][0,2])
              min_area = min(model_area,bb_in_scene_area)
              inter_area = get_intersection_area(model_bb, bb_in_scene[bb])
              union_area = (model_bb[1,0]-model_bb[0,0]) * (model_bb[1,2]-model_bb[0,2])
              union_area = model_area + bb_in_scene_area - inter_area
              if collision_check and inter_area > 0.2 * min_area:
                collision = True
                break
        if not collision:
          furn_node = {}
          furn_node["id"] = "0_" + str(len(bb_in_scene)+1)
          furn_node["type"] = "Object"
          furn_node["valid"] = "1"
          furn_node["modelId"] = best_model
          furn_node["transform"] = torch.transpose(TRS,0,1).flatten().tolist()
          furn_node["bbox"] = furn_bbox
          furn_node["materials"] = []
          nodes.append(furn_node)
          scene_dict["furn_transforms"].append(TRS)
          scene_dict["bb_in_scene"].append(model_bb)
          scene_dict["bb_is_supporter"].append(furniture[f].coarse_category in cat_supporter)
          scene_dict["bb_is_supported"].append(furniture[f].coarse_category in cat_supported)
          scene_dict["bb_is_near_ceil"].append(furniture[f].coarse_category in cat_near_ceil)
          scene_dict["bb_is_chair"].append(furniture[f].coarse_category in cat_chair)
          scene_dict["bb_is_table"].append(furniture[f].coarse_category in cat_table)
          scene_dict["bb_is_sofa"].append(furniture[f].coarse_category == 'sofa')
          scene_dict["bb_is_window"].append(furniture[f].coarse_category == 'window')
          scene_dict["furn_added"].append(True)
          scene_dict["furn_heights"].append(furn_pos_y + model_height)
          break
        elif best_model_index >= np.minimum(len(error_sorted),1)-1:
          nodes.append({})
          scene_dict["furn_transforms"].append(torch.eye(3))
          scene_dict["furn_added"].append(False)
          scene_dict["furn_heights"].append(0)
          has_coll = True
    else: # furniture object is room
      furn_node = {}
      furn_node["id"] =  "0_" + str(f)
      furn_node["type"] = "Room"
      furn_node["valid"] = 1
      furn_node["modelId"] = "fr_0rm_0"
      furn_node["roomTypes"] = ["Bedroom"]
      furn_corners = furn.bbox()
      if orthographic_mode:
        furn_corners = torch.cat([furn_corners[:,[0]],torch.as_tensor([0.0,0.1]).view(2,1),furn_corners[:,[1]]],1)
      else:
        furn_corners = torch.cat([furn_corners[:,[0]],torch.as_tensor([0.0,room_height]).view(2,1),furn_corners[:,[1]]],1)
      furn_bbox = {}
      room_min = furn_corners[0,:]
      room_max = furn_corners[1,:]
      furn_bbox["min"] = room_min.tolist()
      furn_bbox["max"] = room_max.tolist()
      furn_node["bbox"] = furn_bbox
      nodes.append(furn_node)
      scene_dict["furn_added"].append(True)
      scene_dict["furn_heights"].append(torch.as_tensor(room_height))
      scene_dict["furn_transforms"].append(torch.eye(4).flatten().tolist())
      scene_dict["room_min"] = room_min
      scene_dict["room_max"] = room_max
  return nodes, collision, scene_dict, furniture

def reconstruct_3d_scene(full_house_name, nodes, scene_dict, furniture, cat_supporter, cat_supported, path_output_data):
  """
  Reconstructs a 3D scene from a given 2D layout
  """
  orthographic_mode = False
  furn_heights = scene_dict["furn_heights"]
  furn_added = scene_dict["furn_added"]
  furn_transforms = scene_dict["furn_transforms"]
  room_min = scene_dict["room_min"]
  room_max = scene_dict["room_max"]
  bb_in_scene = scene_dict["bb_in_scene"]

  floor_mat_string = "newmtl material_0\nillum 2\nd 1\nNs 30\nKd 0.75 0.75 0.75\nKs 0.06666666666666667 0.06666666666666667 0.06666666666666667"
  wall_mat_string = "newmtl material_0\nillum 2\nd 1\nNs 30\nKd 0.75 0.75 0.75\nKs 0.06666666666666667 0.06666666666666667 0.06666666666666667"
  hole_mat_string = "\nnewmtl material_1\nillum 2\nd 1\nNs 0\nKd 0.0 0.0 0.0\nKs 0.0 0.0 0.0"
  room_height = torch.as_tensor([2.5])
  wall_thick = 0.05

  # adjust height of furniture
  supporters = [furniture[f] for f in range(len(furniture)) if furniture[f].coarse_category in cat_supporter]
  supporter_heights = [furn_heights[f] for f in range(len(furniture)) if furniture[f].coarse_category in cat_supporter]
  supporter_added = [furn_added[f] for f in range(len(furniture)) if furniture[f].coarse_category in cat_supporter]
  for f in range(len(furniture)):
    if furn_added[f]:
      furn = furniture[f]
      if furn.coarse_category in cat_supported:
        height_offset = torch.zeros(1,)
        for s in range(len(supporters)):
          if supporter_added[s]:
            supp = supporters[s]
            if supp.coarse_category == 'shelving' and furn.coarse_category in ['computer','television']:
              continue
            supp_corners = supp.bbox()
            is_inside = (furn.pos[0] >= supp_corners[0,0]) and (furn.pos[1] >= supp_corners[0,1]) and (furn.pos[0] <= supp_corners[1,0]) and (furn.pos[1] <= supp_corners[1,1])
            if is_inside:
              height_offset = torch.maximum(height_offset,supporter_heights[s])
        if (furn.coarse_category == 'television'):
          if (height_offset <= 1e-5):
            min_pos = torch.min(torch.as_tensor([furn.pos[0]**2,furn.pos[1]**2,(furn.pos[0]-furniture[0].dim[0])**2,(furn.pos[1]-furniture[0].dim[1])**2]))
            if min_pos < furn.dim[1]**2: # check if tv close to wall
              height_offset = torch.as_tensor(1.5)
          elif (height_offset > 0.75): # tv stand model too high
            height_offset = torch.as_tensor(0.6)
        if furn.coarse_category == 'indoor_lamp' and height_offset > 1.00:
            height_offset = torch.as_tensor(0.8)
        if height_offset > 0.0:
          TRS = furn_transforms[f]
          T = torch.as_tensor([[1.0,0.0,0.0,0.0], [0.0,1.0,0.0,height_offset], [0.0,0.0,1.0,0.0], [0.0,0.0,0.0,1.0]])
          TRS = torch.matmul(T,TRS)
          nodes[f]["transform"] = torch.transpose(TRS,0,1).flatten().tolist()
          nodes[f]["bbox"]["min"][1] = nodes[f]["bbox"]["min"][1] + height_offset.item()
          nodes[f]["bbox"]["max"][1] = nodes[f]["bbox"]["max"][1] + height_offset.item()

  nodes = [node for node in nodes if len(node.keys()) > 0]
  nodes[0]["nodeIndices"] = [i+1 for i in range(len(bb_in_scene))]
  house_bbox = {}
  house_bbox["min"] = room_min.tolist()
  house_bbox["max"] = room_max.tolist()
  house = {}
  house["version"] = "atek@1.0"
  house["id"] = full_house_name
  house["up"] = [0,1,0]
  house["front"] = [0,0,1]
  house["scaleToMeters"] = 1
  house["levels"] = []
  house["bbox"] = house_bbox
  level = {}
  level["id"] = "0"
  level["bbox"] = house_bbox
  level["nodes"] = nodes
  house["levels"].append(level)

  # create room mesh
  v_floor_bl = torch.as_tensor([room_min[0],room_min[1],room_min[2]]) # 1
  v_floor_br = torch.as_tensor([room_max[0],room_min[1],room_min[2]]) # 2
  v_floor_tr = torch.as_tensor([room_max[0],room_min[1],room_max[2]]) # 3
  v_floor_tl = torch.as_tensor([room_min[0],room_min[1],room_max[2]]) # 4
  v_ceil_bl = torch.as_tensor([room_min[0],room_max[1],room_min[2]]) # 5
  v_ceil_br = torch.as_tensor([room_max[0],room_max[1],room_min[2]]) # 6
  v_ceil_tr = torch.as_tensor([room_max[0],room_max[1],room_max[2]]) # 7
  v_ceil_tl = torch.as_tensor([room_min[0],room_max[1],room_max[2]]) # 8
  v_floor = [v_floor_bl,v_floor_br,v_floor_tr,v_floor_tl]
  v_room = [v_floor_bl,v_floor_br,v_floor_tr,v_floor_tl,v_ceil_bl,v_ceil_br,v_ceil_tr,v_ceil_tl]
  f_floor = [1,4,3,2]
  f_walls = [[1,2,6,5],[2,3,7,6],[3,4,8,7],[4,1,5,8]]

  floor_string = "mtllib fr_0rm_0f.mtl\n" + "o Floor#0_1\n"
  for k in range(len(v_floor)):
    floor_string = floor_string + "v " + str(v_floor[k][0].item()) + " " + str(v_floor[k][1].item()) + " " + str(v_floor[k][2].item()) + "\n"
  floor_string = floor_string + "vt 0.0 0.0\nvt 1.0 0.0\nvt 1.0 1.0\nvt 0.0 1.0\n"
  floor_string = floor_string + "usemtl material_0\n"
  floor_string = floor_string + "f " + str(f_floor[0]) + "/1 " + str(f_floor[1]) + "/4 " + str(f_floor[2]) + "/3 " + str(f_floor[3]) + "/2\n"

  walls_string = "mtllib fr_0rm_0w.mtl\n"
  for w in range(len(f_walls)):
    walls_string = walls_string + "o Wall#0_" + str(w+1) + "\n"
    f_wall = f_walls[w]
    offset = 2 * w * len(f_wall)
    wall_offset = wall_thick * torch.as_tensor([np.sin(w*np.pi/2), 0.0, -np.cos(w*np.pi/2)]).view(1,3)
    for k in range(len(f_wall)):
      walls_string = walls_string + "v " + str(v_room[f_wall[k]-1][0].item()) + " " + str(v_room[f_wall[k]-1][1].item()) + " " + str(v_room[f_wall[k]-1][2].item()) + "\n"
    for k in range(len(f_wall)):
      signs = [-1,1,1,-1]
      v_room_off = torch.stack(v_room,0) + wall_offset + signs[k] * wall_thick * torch.as_tensor([np.cos(w*np.pi/2), 0.0, np.sin(w*np.pi/2)]).view(1,3)
      walls_string = walls_string + "v " + str(v_room_off[f_wall[k]-1][0].item()) + " " + str(v_room_off[f_wall[k]-1][1].item()) + " " + str(v_room_off[f_wall[k]-1][2].item()) + "\n"
    walls_string = walls_string + "vt 0.0 0.0\nvt 1.0 0.0\nvt 1.0 1.0\nvt 0.0 1.0\n"
    walls_string = walls_string + "usemtl material_0\n"
    walls_string = walls_string + "f " + str(1+offset) + "/1 " + str(2+offset) + "/2 " + str(3+offset) + "/3 " + str(4+offset) + "/4\n"
    walls_string = walls_string + "f " + str(4+offset) + " " + str(3+offset) + " " + str(7+offset) + " " + str(8+offset) + "\n"
    walls_string = walls_string + "f " + str(5+offset) + " " + str(8+offset) + " " + str(7+offset) + " " + str(6+offset) + "\n"

  # add 'holes' for the doors
  room_center = 0.5 * torch.sum(furniture[0].bbox(), 0)
  holes_string = ""
  h = 0
  # for f in range(len(furniture)):
    # furn = furniture[f]
    # if furn.coarse_category == 'door':
      # h = h + 1
      # holes_string = holes_string + "o Hole#0_" + str(h) + "\n"
      # door_vec = (furn.pos - room_center) / torch.linalg.norm(furn.pos - room_center)
      # dir_sign = -1 * torch.sign(torch.dot(door_vec,furn.world_front()))
      # door_dims = furn.dim
      # door_front = furn.world_front() * 0.52 * door_dims[1]
      # door_side = torch.as_tensor([furn.world_front()[1],-furn.world_front()[0]]) * 0.50 * door_dims[0]
      # door_pos = furn.pos + door_front * dir_sign
      # v_hole_bl = torch.as_tensor([door_pos[0] - door_side[0], 0.0, door_pos[1] - door_side[1]])
      # v_hole_br = torch.as_tensor([door_pos[0] + door_side[0], 0.0, door_pos[1] + door_side[1]])
      # v_hole_tl = torch.as_tensor([door_pos[0] - door_side[0], furn_heights[f], door_pos[1] - door_side[1]])
      # v_hole_tr = torch.as_tensor([door_pos[0] + door_side[0], furn_heights[f], door_pos[1] + door_side[1]])
      # v_hole = [v_hole_bl,v_hole_br,v_hole_tr,v_hole_tl]
      # if dir_sign < 0:
        # f_hole = [1,4,3,2]
      # else:
        # f_hole = [1,2,3,4]
      # offset = 2 * len(f_wall)**2 + (h-1) * len(v_hole)
      # for k in range(len(v_hole)):
        # holes_string = holes_string + "v " + str(v_hole[k][0].item()) + " " + str(v_hole[k][1].item()) + " " + str(v_hole[k][2].item()) + "\n"
      # holes_string = holes_string + "vt 0.0 0.0\nvt 1.0 0.0\nvt 1.0 1.0\nvt 0.0 1.0\n"
      # holes_string = holes_string + "usemtl material_1\n"
      # holes_string = holes_string + "f " + str(offset+f_hole[0]) + "/17 " + str(offset+f_hole[1]) + "/18 " + str(offset+f_hole[2]) + "/19 " + str(offset+f_hole[3]) + "/20\n"

  if not os.path.isdir(path_output_data + "house/"):
    os.mkdir(path_output_data + "house/")
  if not os.path.isdir(path_output_data + "room/"):
    os.mkdir(path_output_data + "room/")
  if not os.path.isdir(path_output_data + "house/" + full_house_name):
    os.mkdir(path_output_data + "house/" + full_house_name)
  with open(path_output_data + "house/" + full_house_name + "/house.json", "w") as f:
    json.dump(house, f, indent=4)
  if not os.path.isdir(path_output_data + "room/" + full_house_name):
    os.mkdir(path_output_data + "room/" + full_house_name)
  with open(path_output_data + "room/" + full_house_name + "/fr_0rm_0f.obj", 'w') as f:
    f.write(floor_string)
  with open(path_output_data + "room/" + full_house_name + "/fr_0rm_0f.mtl", 'w') as f:
    f.write(floor_mat_string)
  with open(path_output_data + "room/" + full_house_name + "/fr_0rm_0w.mtl", 'w') as f:
    if orthographic_mode:
      f.write(wall_mat_string)
    else:
      f.write(wall_mat_string + hole_mat_string)
  with open(path_output_data + "room/" + full_house_name + "/fr_0rm_0w.obj", 'w') as f:
    if orthographic_mode:
      f.write(walls_string)
    else:
      f.write(walls_string + holes_string)