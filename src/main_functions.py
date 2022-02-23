import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
from torch.nn import functional as F
import transformers

from src.modeling_gpt2_custom import GPT2LMHeadModelCustom

def gaussian1d(mu,sigma,res,device='cpu'):
  """
  Creates a discrete non-normalized gaussian PDF in the range [0,res-1]

  Parameters:
    mu (float): center of the normal distribution
    sigma (float): variance of the normal distribution
    res (int): determines the range [0,res-1] in which the normal distribution is evaluated
    device (String): device of the returned tensor
  Returns:
    1D-tensor (float): tensor with shape [res] containing the evaluated PDF
  """
  mu = torch.as_tensor(mu).view(-1,1)
  x = torch.linspace(0, res-1, res,device=device).view(1,-1)
  return torch.exp(-0.5*((x-mu)/sigma)**2) #* 1/(s*np.sqrt(2*np.pi)) 

def glare_par(pos,light,grid,all_pairs=True):
  """
  Evaluates the glare ergonomic score
  """
  if all_pairs:
    v = torch.unsqueeze(grid,1)-torch.unsqueeze(pos,2)
  else:
    v = (grid - pos).unsqueeze(2)
  v = v / torch.linalg.norm(v,dim=-1,keepdim=True)
  lp = torch.unsqueeze(light,1)-torch.unsqueeze(pos,2)
  lp = lp / torch.linalg.norm(lp,dim=-1,keepdim=True)
  vdotlp = torch.unsqueeze(v,2) * torch.unsqueeze(lp,3)
  E = ((torch.sum(vdotlp,-1)+1.0)/2.0)**4
  sm = torch.nn.Softmax(2)
  E = torch.sum(E*sm(E*10.0),2)
  return E

def lighting_par(pos,light,grid,all_pairs=True):
  """
  Evaluates the lighting ergonomic score
  """
  if all_pairs:
    v = torch.unsqueeze(grid,1)-torch.unsqueeze(pos,2)
    v = v / torch.linalg.norm(v,dim=-1,keepdim=True)
    lg = torch.unsqueeze(grid,1)-torch.unsqueeze(light,2)
    lg = lg / torch.linalg.norm(lg,dim=-1,keepdim=True)
    vdotlg = torch.unsqueeze(v,2) * torch.unsqueeze(lg,1)
  else:
    v = (grid - pos).unsqueeze(2)
    v = v / torch.linalg.norm(v,dim=-1,keepdim=True)
    lg = torch.unsqueeze(grid,2)-torch.unsqueeze(light,1)
    lg = lg / torch.linalg.norm(lg,dim=-1,keepdim=True)
    vdotlg = torch.unsqueeze(v,2) * torch.unsqueeze(lg,3)
  E = ((torch.sum(vdotlg,-1)+1.0)/2.0)
  E = (1.0-E)**4
  sm = torch.nn.Softmin(2)
  E = torch.sum(E*sm(E*10.0),2)
  return E

def visibility_par(pos,v,grid):
  """
  Evaluates the visibility ergonomic score
  """
  pg = torch.unsqueeze(grid,1)-torch.unsqueeze(pos,2)
  pg = pg / torch.linalg.norm(pg,dim=-1,keepdim=True)
  v = v / torch.linalg.norm(v,dim=-1,keepdim=True)
  vdotpg = torch.unsqueeze(v,2) * pg#torch.unsqueeze(pg,1)
  E = 1.0-((torch.sum(vdotpg,-1)+1.0)/2.0)**2
  return E

def reach_par(pos,grid,easy_reach=0.8,dropoff=15.0):
  """
  Evaluates the reach ergonomic score
  """
  pg = torch.unsqueeze(grid,1)-torch.unsqueeze(pos,2)
  pg = torch.linalg.norm(pg,dim=-1,keepdim=False)
  E = 1.0 / (1.0 + torch.exp(-dropoff * (pg - easy_reach)))
  return E

class Furniture():
  """
  Represents a furniture object
  """
  def __init__(self, sequence=None, dict_int2cat=None,keep_node=False,device='cpu'):
    """
    Creates a new furniture object
    """
    if sequence is not None and dict_int2cat is not None:
      self.from_sequence(sequence,dict_int2cat,device=device)
    else:
      self.node = None
      self.fine_category = None       
      self.coarse_category = None
      self.final_category = None
      self.pos = None
      self.dim = None
      self.rot = None
      self.front = None
    self.device=device

  def from_sequence(self,sequence,dict_int2cat,device='cpu'):
    """
    Creates a new furniture object from a sequence
    """
    self.node = None
    self.fine_category = dict_int2cat[str(int(sequence[0]))]     
    self.coarse_category = self.fine_category
    self.final_category = self.fine_category
    self.pos = sequence[1:3].to(device)
    self.dim = sequence[3:5].to(device)
    self.rot = sequence[5].view(-1,).to(device)
    self.front = torch.as_tensor([0.0,1.0]).to(device)

  @staticmethod
  def from_room(room,device='cpu'):
    """
    Creates a new room-type furniture object from a room-dict
    """
    room_obj = Furniture()
    room_obj.node = None
    room_obj.fine_category = 'room'   
    room_obj.coarse_category = 'room'
    room_obj.final_category = 'room'
    min_pos = torch.as_tensor(room.bbox['min'])[[0,2]].to(device)
    max_pos = torch.as_tensor(room.bbox['max'])[[0,2]].to(device)
    room_obj.pos = 0.5 * (min_pos + max_pos)
    room_obj.dim = max_pos - min_pos
    room_obj.rot = torch.as_tensor([0.0]).to(device)
    room_obj.front = torch.as_tensor([0.0,1.0]).to(device)
    return room_obj

  def world_front(self):
    """
    Returns the orientation of the furniture object in world coordinates
    """
    rotmat = torch.stack([torch.cat([torch.cos(self.rot),-torch.sin(self.rot)]),torch.cat([torch.sin(self.rot),torch.cos(self.rot)])],0)
    return torch.matmul(rotmat,self.front.view(2,-1).to(self.device)).squeeze()

  def to_sequence(self,dict_cat2int):
    """
    Creates a sequence from the furniture object
    """
    cat_int = dict_cat2int[self.coarse_category]
    if self.fine_category == "chandelier":
      cat_int = dict_cat2int[self.fine_category]
    return torch.cat([torch.as_tensor([cat_int]),self.pos,self.dim,self.rot.view(1,)])

  def bbox(self):
    """
    Returns the bottom left and top right oriented bounding box corners of the furniture object
    """
    front = self.world_front()
    side = torch.as_tensor([-front[1].clone(),front[0].clone()],device=front.device) * 0.5 * self.dim[0]
    front = front * 0.5 * self.dim[1]
    corners = torch.stack([self.pos + front + side, self.pos + front - side, self.pos - front + side, self.pos - front - side],0)
    min_corner = torch.min(corners,0)[0]
    max_corner = torch.max(corners,0)[0]
    return torch.stack([min_corner,max_corner],0)

  def four_corners(self):
    """
    Returns the four oriented bounding box corners of the furniture object
    """
    front = self.world_front()
    side = torch.as_tensor([-front[1].clone(),front[0].clone()],device=front.device) * 0.5 * self.dim[0]
    front = front * 0.5 * self.dim[1]
    corners = torch.stack([self.pos + front + side, self.pos + front - side, self.pos - front + side, self.pos - front - side],0)
    return corners

  def nine_samples(self):
    """
    Returns the center, edge-midpoints and corners of the oriented bounding box of the furniture object
    """
    front = self.world_front()
    side = torch.as_tensor([-front[1].clone(),front[0].clone()],device=front.device) * 0.5 * self.dim[0]
    front = front * 0.5 * self.dim[1]
    corners = torch.stack([self.pos, self.pos + front, self.pos - side, self.pos - front, self.pos + side,
                           self.pos + front + side, self.pos + front - side, self.pos - front + side, self.pos - front - side],0)
    return corners

  def __repr__(self):
    """
    Returns object category
    """
    return self.coarse_category

def filter_furniture(furniture,cat_valid,centering=False):
  """
  Filters out furniture objects with invalid categories and optionally centers the furniture
  """
  filtered_furniture = []
  for i in range(len(furniture)):
    if furniture[i].coarse_category in cat_valid:
      filtered_furniture.append(furniture[i])
  if len(filtered_furniture) > 0 and centering:
    filtered_furniture = center_furniture(filtered_furniture)
  return filtered_furniture

def center_furniture(furniture):
  """
  Centers the furniture such that the mean of all furniture positions is at [0,0]
  """
  cat_windoor = ['window','door']
  is_windoor = torch.stack([torch.as_tensor(f.coarse_category in cat_windoor) for f in furniture],0)
  pos = torch.stack([f.pos for f in furniture],0)
  dim = torch.stack([f.dim for f in furniture],0)
  front = torch.stack([f.world_front() for f in furniture],0) 
  side = torch.stack([-front[:,1].clone(),front[:,0].clone()],1) * 0.5 * dim[:,0].view(-1,1)
  front = front * 0.5 * dim[:,1].view(-1,1)
  corners = torch.stack([pos + front + side, pos + front - side, pos - front + side, pos - front - side],2)
  corners[is_windoor,:,2:4] = corners[is_windoor,:,0:2] # for doors and windows only the front is inside the room
  room_min = torch.min(torch.min(corners,2)[0],0)[0]
  room_max = torch.max(torch.max(corners,2)[0],0)[0]
  center = 0.5*(room_min+room_max)
  for f in furniture:
    f.pos = f.pos - center
  return furniture

def get_predictions(labels, logits,minvalue_dict,maxvalue_dict,res,device='cpu'):
  """
  Computes the interpolated value of a predicted token based on its neighborhood
  """
  logits = F.softmax(logits,-1)
  pred_ind = torch.argmax(logits,-1)
  gauss_weights = gaussian1d(pred_ind,8.0,res,device=device)
  row_ind = torch.linspace(0,logits.size(0)-1,logits.size(0),dtype=torch.long,device=device)
  col_ind = torch.linspace(0,logits.size(1)-2,logits.size(1)-1,dtype=torch.long,device=device)
  interp_val = torch.sum(col_ind * logits[row_ind,:-1] * gauss_weights,1)
  prob_sum = torch.sum(logits[row_ind,:-1] * gauss_weights,1)
  interp_val[prob_sum > 0.0] = interp_val[prob_sum > 0.0] / prob_sum[prob_sum > 0.0]
  interp_val = torch.cat([torch.zeros(1,device=device),interp_val[:-2]]).view(-1,6)
  label_mat = labels[:-1].view(-1,6).clone()
  return interp_val, label_mat

def score_read_book(labels, minvalue_dict,maxvalue_dict,res,dict_cat2fun,use_log=True,device='cpu'):
  """
  Evaluates the Read Book activity for the given layout
  """
  label_mat = labels.clone()
  ind_posdimrot = [1,2,5]
  cat_light = dict_cat2fun["light"]
  cat_light_glare = dict_cat2fun["glare"]
  cat_sit = dict_cat2fun["sit"]
  cat_sitlight = torch.concat([cat_light,cat_sit],dim=1)
  is_sitlight = torch.sum(torch.eq(cat_sitlight,label_mat[:,[0]]),-1) > 0
  is_valid = torch.sum(label_mat > (res-0.9),1) < 1.0
  label_mat = label_mat[torch.logical_and(is_sitlight, is_valid),:]
  
  is_sit = torch.sum(torch.eq(cat_sit,label_mat[:,[0]]),-1) > 0
  is_light = torch.sum(torch.eq(cat_light,label_mat[:,[0]]),-1) > 0
  is_light_glare = torch.sum(torch.eq(cat_light_glare,label_mat[:,[0]]),-1) > 0
  if torch.sum(is_sit) < 1 or torch.sum(is_light) < 1:
    return -2.0*torch.ones(1,).to(device)

  pos_lights = label_mat[is_light,1:3].unsqueeze(0)
  pos_lights_glare = label_mat[is_light_glare,1:3].unsqueeze(0)
  pos_sit = label_mat[is_sit,1:3].unsqueeze(0)
  dim_sit = label_mat[is_sit,3:5].unsqueeze(0)
  rot_sit = label_mat[is_sit,5:6].unsqueeze(0)
  dir_sit = torch.cat([-torch.sin(rot_sit),torch.cos(rot_sit)],-1)
  pos_sit = pos_sit - dir_sit * 0.25 * dim_sit
  pos_books = pos_sit + dir_sit * 0.2
  epsilon = torch.exp(torch.as_tensor([-5])).to(device)
  e_glare = glare_par(pos_sit,pos_lights_glare,pos_books,all_pairs=False)
  e_lighting = lighting_par(pos_sit,pos_lights,pos_books,all_pairs=False)
  if use_log:
    e_glare = -torch.log(1.0 + epsilon - e_glare)
    e_lighting = -torch.log(1.0 + epsilon - e_lighting)
  e_total = 0.5 * e_glare + 0.5 * e_lighting
  e_total = e_total.flatten(start_dim=-2)
  return torch.sum(e_total * F.softmin(e_total*10.0,-1),-1)

def eval_read_book(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun,use_log=True,device='cpu'):  
  """
  Evaluates the Read Book activity for each predicted token value
  """
  ind_posdimrot = [1,2,5]
  cat_light = dict_cat2fun["light"]
  cat_light_glare = dict_cat2fun["glare"]
  cat_sit = dict_cat2fun["sit"]
  cat_sitlight = torch.concat([cat_light,cat_sit],dim=1)
  is_sitlight = torch.sum(torch.eq(cat_sitlight,label_mat[:,[0]]),-1) > 0
  interp_val = interp_val[is_sitlight,:]
  label_mat = label_mat[is_sitlight,:]

  is_sit = torch.sum(torch.eq(cat_sit,label_mat[:,[0]]),-1) > 0
  is_light = torch.sum(torch.eq(cat_light,label_mat[:,[0]]),-1) > 0
  is_light_glare = torch.sum(torch.eq(cat_light_glare,label_mat[:,[0]]),-1) > 0
  if torch.sum(is_sit) < 1 or torch.sum(is_light) < 1:
    return -2.0*torch.ones(1,).to(device)
  interp_val = interp_val[:,ind_posdimrot]
  value_mat = label_mat[:,ind_posdimrot].clone()

  # replace gt values with predicted values
  n_rows = interp_val.size(0)
  n_cols = interp_val.size(1)
  n_elem = n_rows * n_cols
  value_mat = value_mat.repeat(n_elem,1,1)
  elem_ind = torch.linspace(0,n_elem-1,n_elem,dtype=torch.long,device=device)
  col_ind = torch.linspace(0,n_cols-1,n_cols,dtype=torch.long,device=device).repeat_interleave(n_rows)
  row_ind = torch.linspace(0,n_rows-1,n_rows,dtype=torch.long,device=device).repeat(n_cols)
  value_mat[elem_ind,row_ind,col_ind] = interp_val[row_ind,col_ind]
  
  pos_lights = value_mat[:,is_light,0:2]
  pos_lights_glare = value_mat[:,is_light_glare,0:2]
  pos_sit = value_mat[:,is_sit,0:2]
  dim_sit = label_mat[is_sit,3:5].view(1,-1,2)
  rot_sit = value_mat[:,is_sit,2:3]
  dir_sit = torch.cat([-torch.sin(rot_sit),torch.cos(rot_sit)],-1)
  pos_sit = pos_sit - dir_sit * 0.25 * dim_sit
  pos_books = pos_sit + dir_sit * 0.2
  epsilon = torch.exp(torch.as_tensor([-5])).to(device)
  e_glare = glare_par(pos_sit,pos_lights_glare,pos_books,all_pairs=False)
  e_lighting = lighting_par(pos_sit,pos_lights,pos_books,all_pairs=False)
  if use_log:
    e_glare = -torch.log(1.0 + epsilon - e_glare)
    e_lighting = -torch.log(1.0 + epsilon - e_lighting)
  e_total = 0.5 * e_glare + 0.5 * e_lighting
  e_total = e_total.flatten(start_dim=-2)
  return torch.sum(e_total * F.softmin(e_total*10.0,-1),-1)

def score_watch_tv(labels, minvalue_dict,maxvalue_dict,res,dict_cat2fun,use_log=True,device='cpu'):
  """
  Evaluates the Watch TV activity for the given layout
  """
  label_mat = labels.clone()
  ind_posdimrot = [1,2,5]
  cat_light_glare = dict_cat2fun["glare"]
  cat_sit = dict_cat2fun["sit"]
  cat_tv = dict_cat2fun["tv"]
  cat_sitlighttv = torch.concat([cat_light_glare,cat_sit,cat_tv],dim=1)
  is_sitlighttv = torch.sum(torch.eq(cat_sitlighttv,label_mat[:,[0]]),-1) > 0
  is_valid = torch.sum(label_mat > (res-0.9),1) < 1.0
  label_mat = label_mat[torch.logical_and(is_sitlighttv, is_valid),:]

  is_tv = torch.sum(torch.eq(cat_tv,label_mat[:,[0]]),-1) > 0
  is_sit = torch.sum(torch.eq(cat_sit,label_mat[:,[0]]),-1) > 0
  is_light_glare = torch.sum(torch.eq(cat_light_glare,label_mat[:,[0]]),-1) > 0
  if torch.sum(is_sit) < 1 or torch.sum(is_tv) < 1 or torch.sum(is_light_glare) < 1:
    return -2.0*torch.ones(1,).to(device)
  
  pos_tv = label_mat[is_tv,1:3].unsqueeze(0)
  pos_lights_glare = label_mat[is_light_glare,1:3].unsqueeze(0)
  pos_sit = label_mat[is_sit,1:3].unsqueeze(0)
  dim_sit = label_mat[is_sit,3:5].unsqueeze(0)
  rot_sit = label_mat[is_sit,5:6].unsqueeze(0)
  dir_sit = torch.cat([-torch.sin(rot_sit),torch.cos(rot_sit)],-1)
  epsilon = torch.exp(torch.as_tensor([-5])).to(device)
  e_glare = glare_par(pos_sit,pos_lights_glare,pos_tv,all_pairs=True)
  e_vis = visibility_par(pos_sit,dir_sit,pos_tv)
  if use_log:
    e_glare = -torch.log(1.0 + epsilon - e_glare)
    e_vis = -torch.log(1.0 + epsilon - e_vis)
  e_total = 0.5 * e_glare + 0.5 * e_vis
  e_total = e_total.flatten(start_dim=-2)
  return torch.sum(e_total * F.softmin(e_total*10.0,-1),-1)

def eval_watch_tv(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun,use_log=True,device='cpu'):
  """
  Evaluates the Watch TV activity for each predicted token value
  """
  ind_posdimrot = [1,2,5]
  cat_light_glare = dict_cat2fun["glare"]
  cat_sit = dict_cat2fun["sit"]
  cat_tv = dict_cat2fun["tv"]
  cat_sitlighttv = torch.concat([cat_light_glare,cat_sit,cat_tv],dim=1)
  is_sitlighttv = torch.sum(torch.eq(cat_sitlighttv,label_mat[:,[0]]),-1) > 0
  interp_val = interp_val[is_sitlighttv,:]
  label_mat = label_mat[is_sitlighttv,:]

  is_sit = torch.sum(torch.eq(cat_sit,label_mat[:,[0]]),-1) > 0
  is_light_glare = torch.sum(torch.eq(cat_light_glare,label_mat[:,[0]]),-1) > 0
  is_tv = torch.sum(torch.eq(cat_tv,label_mat[:,[0]]),-1) > 0
  if torch.sum(is_sit) < 1 or torch.sum(is_tv) < 1 or torch.sum(is_light_glare) < 1:
    return -2.0*torch.ones(1,).to(device)
  interp_val = interp_val[:,ind_posdimrot]
  value_mat = label_mat[:,ind_posdimrot].clone()

  # replace gt values with predicted values
  n_rows = interp_val.size(0)
  n_cols = interp_val.size(1)
  n_elem = n_rows * n_cols
  value_mat = value_mat.repeat(n_elem,1,1)
  elem_ind = torch.linspace(0,n_elem-1,n_elem,dtype=torch.long,device=device)
  col_ind = torch.linspace(0,n_cols-1,n_cols,dtype=torch.long,device=device).repeat_interleave(n_rows)
  row_ind = torch.linspace(0,n_rows-1,n_rows,dtype=torch.long,device=device).repeat(n_cols)
  value_mat[elem_ind,row_ind,col_ind] = interp_val[row_ind,col_ind]
  
  pos_tv = value_mat[:,is_tv,0:2]
  pos_lights_glare = value_mat[:,is_light_glare,0:2]
  pos_sit = value_mat[:,is_sit,0:2]
  rot_sit = value_mat[:,is_sit,2:3]
  dir_sit = torch.cat([-torch.sin(rot_sit),torch.cos(rot_sit)],-1)
  epsilon = torch.exp(torch.as_tensor([-5])).to(device)
  e_glare = glare_par(pos_sit,pos_lights_glare,pos_tv,all_pairs=True)
  e_vis = visibility_par(pos_sit,dir_sit,pos_tv)
  if use_log:
    e_glare = -torch.log(1.0 + epsilon - e_glare)
    e_vis = -torch.log(1.0 + epsilon - e_vis)
  e_total = 0.5 * e_glare + 0.5 * e_vis
  e_total = e_total.flatten(start_dim=-2)
  return torch.sum(e_total * F.softmin(e_total*10.0,-1),-1)

def score_use_computer(labels, minvalue_dict,maxvalue_dict,res,dict_cat2fun,use_log=True,device='cpu'):
  """
  Evaluates the Use Computer activity for the given layout
  """
  label_mat = labels.clone()
  ind_posdimrot = [1,2,5]
  cat_light_glare = dict_cat2fun["glare"]
  cat_sit = dict_cat2fun["sit"]
  cat_comp = dict_cat2fun["computer"]
  cat_sitlightcomp = torch.concat([cat_light_glare,cat_sit,cat_comp],dim=1)
  is_sitlightcomp = torch.sum(torch.eq(cat_sitlightcomp,label_mat[:,[0]]),-1) > 0
  is_valid = torch.sum(label_mat > (res-0.9),1) < 1.0
  label_mat = label_mat[torch.logical_and(is_sitlightcomp, is_valid),:]

  is_comp = torch.sum(torch.eq(cat_comp,label_mat[:,[0]]),-1) > 0
  is_sit = torch.sum(torch.eq(cat_sit,label_mat[:,[0]]),-1) > 0
  is_light_glare = torch.sum(torch.eq(cat_light_glare,label_mat[:,[0]]),-1) > 0
  if torch.sum(is_sit) < 1 or torch.sum(is_comp) < 1 or torch.sum(is_light_glare) < 1:
    return -2.0*torch.ones(1,).to(device)
  
  pos_comp = label_mat[is_comp,1:3].unsqueeze(0)
  pos_lights_glare = label_mat[is_light_glare,1:3].unsqueeze(0)
  pos_sit = label_mat[is_sit,1:3].unsqueeze(0)
  dim_sit = label_mat[is_sit,3:5].unsqueeze(0)
  rot_sit = label_mat[is_sit,5:6].unsqueeze(0)
  dir_sit = torch.cat([-torch.sin(rot_sit),torch.cos(rot_sit)],-1)
  epsilon = torch.exp(torch.as_tensor([-5])).to(device)
  e_glare = glare_par(pos_sit,pos_lights_glare,pos_comp,all_pairs=True)
  e_vis = visibility_par(pos_sit,dir_sit,pos_comp)
  e_reach = reach_par(pos_sit,pos_comp)
  if use_log:
    e_glare = -torch.log(1.0 + epsilon - e_glare)
    e_vis = -torch.log(1.0 + epsilon - e_vis)
    e_reach = -torch.log(1.0 + epsilon - e_reach)
  e_total = e_glare / 3.0 + e_vis / 3.0 + e_reach / 3.0
  e_total = e_total.flatten(start_dim=-2)
  return torch.sum(e_total * F.softmin(e_total*10.0,-1),-1)

def eval_use_computer(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun,use_log=True,device='cpu'):  
  """
  Evaluates the Use Computer activity for each predicted token value
  """
  ind_posdimrot = [1,2,5]
  cat_light_glare = dict_cat2fun["glare"]
  cat_sit = dict_cat2fun["sit"]
  cat_comp = dict_cat2fun["computer"]
  cat_sitlightcomp = torch.concat([cat_light_glare,cat_sit,cat_comp],dim=1)
  is_sitlightcomp = torch.sum(torch.eq(cat_sitlightcomp,label_mat[:,[0]]),-1) > 0
  interp_val = interp_val[is_sitlightcomp,:]
  label_mat = label_mat[is_sitlightcomp,:]

  is_sit = torch.sum(torch.eq(cat_sit,label_mat[:,[0]]),-1) > 0
  is_light_glare = torch.sum(torch.eq(cat_light_glare,label_mat[:,[0]]),-1) > 0
  is_comp = torch.sum(torch.eq(cat_comp,label_mat[:,[0]]),-1) > 0
  if torch.sum(is_sit) < 1 or torch.sum(is_comp) < 1 or torch.sum(is_light_glare) < 1:
    return -2.0*torch.ones(1,).to(device)
  interp_val = interp_val[:,ind_posdimrot]
  value_mat = label_mat[:,ind_posdimrot].clone()

  # replace gt values with predicted values
  n_rows = interp_val.size(0)
  n_cols = interp_val.size(1)
  n_elem = n_rows * n_cols
  value_mat = value_mat.repeat(n_elem,1,1)
  elem_ind = torch.linspace(0,n_elem-1,n_elem,dtype=torch.long,device=device)
  col_ind = torch.linspace(0,n_cols-1,n_cols,dtype=torch.long,device=device).repeat_interleave(n_rows)
  row_ind = torch.linspace(0,n_rows-1,n_rows,dtype=torch.long,device=device).repeat(n_cols)
  value_mat[elem_ind,row_ind,col_ind] = interp_val[row_ind,col_ind]
  
  pos_comp = value_mat[:,is_comp,0:2]
  pos_lights_glare = value_mat[:,is_light_glare,0:2]
  pos_sit = value_mat[:,is_sit,0:2]
  rot_sit = value_mat[:,is_sit,2:3]
  dir_sit = torch.cat([-torch.sin(rot_sit),torch.cos(rot_sit)],-1)
  epsilon = torch.exp(torch.as_tensor([-5])).to(device)
  e_glare = glare_par(pos_sit,pos_lights_glare,pos_comp,all_pairs=True)
  e_vis = visibility_par(pos_sit,dir_sit,pos_comp)
  e_reach = reach_par(pos_sit,pos_comp)
  if use_log:
    e_glare = -torch.log(1.0 + epsilon - e_glare)
    e_vis = -torch.log(1.0 + epsilon - e_vis)
    e_reach = -torch.log(1.0 + epsilon - e_reach)
  e_total = e_glare / 3.0 + e_vis / 3.0 + e_reach / 3.0
  e_total = e_total.flatten(start_dim=-2)
  return torch.sum(e_total * F.softmin(e_total*10.0,-1),-1)

def score_work_table(labels, minvalue_dict,maxvalue_dict,res,dict_cat2fun,use_log=True,device='cpu'):
  """
  Evaluates the Work at Table activity for the given layout
  """
  label_mat = labels.clone()
  ind_posdimrot = [1,2,5]
  cat_light = dict_cat2fun["light"]
  cat_sit = dict_cat2fun["sit"]
  cat_table = dict_cat2fun["table"]
  cat_sitlighttable = torch.concat([cat_light,cat_sit,cat_table],dim=1)
  is_sitlighttable = torch.sum(torch.eq(cat_sitlighttable,label_mat[:,[0]]),-1) > 0
  is_valid = torch.sum(label_mat > (res-0.9),1) < 1.0
  label_mat = label_mat[torch.logical_and(is_sitlighttable, is_valid),:]

  is_table = torch.sum(torch.eq(cat_table,label_mat[:,[0]]),-1) > 0
  is_sit = torch.sum(torch.eq(cat_sit,label_mat[:,[0]]),-1) > 0
  is_light = torch.sum(torch.eq(cat_light,label_mat[:,[0]]),-1) > 0
  if torch.sum(is_sit) < 1 or torch.sum(is_table) < 1 or torch.sum(is_light) < 1:
    return -2.0*torch.ones(1,).to(device)
  
  pos_table = label_mat[is_table,1:3].unsqueeze(0)
  pos_lights = label_mat[is_light,1:3].unsqueeze(0)
  pos_sit = label_mat[is_sit,1:3].unsqueeze(0)
  dim_sit = label_mat[is_sit,3:5].unsqueeze(0)
  rot_sit = label_mat[is_sit,5:6].unsqueeze(0)
  dir_sit = torch.cat([-torch.sin(rot_sit),torch.cos(rot_sit)],-1)
  epsilon = torch.exp(torch.as_tensor([-5])).to(device)
  e_lighting = lighting_par(pos_sit,pos_lights,pos_table)
  e_vis = visibility_par(pos_sit,dir_sit,pos_table)
  e_reach = reach_par(pos_sit,pos_table, easy_reach=1.0)
  if use_log:
    e_lighting = -torch.log(1.0 + epsilon - e_lighting)
    e_reach = -torch.log(1.0 + epsilon - e_reach)
    e_vis = -torch.log(1.0 + epsilon - e_vis)
  e_total = e_lighting / 3.0 + e_vis / 3.0 + e_reach / 3.0
  e_total = e_total.flatten(start_dim=-2)
  return torch.sum(e_total * F.softmin(e_total*10.0,-1),-1)

def eval_work_table(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun,use_log=True,device='cpu'):
  """
  Evaluates the Work at Table activity for each predicted token value
  """
  ind_posdimrot = [1,2,5]
  cat_light = dict_cat2fun["light"]
  cat_sit = dict_cat2fun["sit"]
  cat_table = dict_cat2fun["table"]
  cat_sitlighttable = torch.concat([cat_light,cat_sit,cat_table],dim=1)
  is_sitlighttable = torch.sum(torch.eq(cat_sitlighttable,label_mat[:,[0]]),-1) > 0
  is_valid = torch.sum(interp_val > (res-0.9),1) + torch.sum(label_mat > (res-0.9),1) < 1.0
  interp_val = interp_val[is_sitlighttable,:]
  label_mat = label_mat[is_sitlighttable,:]

  is_sit = torch.sum(torch.eq(cat_sit,label_mat[:,[0]]),-1) > 0
  is_light = torch.sum(torch.eq(cat_light,label_mat[:,[0]]),-1) > 0
  is_table = torch.sum(torch.eq(cat_table,label_mat[:,[0]]),-1) > 0
  if torch.sum(is_sit) < 1 or torch.sum(is_table) < 1 or torch.sum(is_light) < 1:
    return -2.0*torch.ones(1,).to(device)
  interp_val = interp_val[:,ind_posdimrot]
  value_mat = label_mat[:,ind_posdimrot].clone()

  # replace gt values with predicted values
  n_rows = interp_val.size(0)
  n_cols = interp_val.size(1)
  n_elem = n_rows * n_cols
  value_mat = value_mat.repeat(n_elem,1,1)
  elem_ind = torch.linspace(0,n_elem-1,n_elem,dtype=torch.long,device=device)
  col_ind = torch.linspace(0,n_cols-1,n_cols,dtype=torch.long,device=device).repeat_interleave(n_rows)
  row_ind = torch.linspace(0,n_rows-1,n_rows,dtype=torch.long,device=device).repeat(n_cols)
  value_mat[elem_ind,row_ind,col_ind] = interp_val[row_ind,col_ind]
  
  pos_table = value_mat[:,is_table,0:2]
  pos_lights = value_mat[:,is_light,0:2]
  pos_sit = value_mat[:,is_sit,0:2]
  rot_sit = value_mat[:,is_sit,2:3]
  dir_sit = torch.cat([-torch.sin(rot_sit),torch.cos(rot_sit)],-1)
  epsilon = torch.exp(torch.as_tensor([-5])).to(device)
  e_lighting = lighting_par(pos_sit,pos_lights,pos_table)
  e_vis = visibility_par(pos_sit,dir_sit,pos_table)
  e_reach = reach_par(pos_sit,pos_table, easy_reach=1.0)
  if use_log:
    e_lighting = -torch.log(1.0 + epsilon - e_lighting)
    e_reach = -torch.log(1.0 + epsilon - e_reach)
    e_vis = -torch.log(1.0 + epsilon - e_vis)
  e_total = e_lighting / 3.0 + e_vis / 3.0 + e_reach / 3.0
  e_total = e_total.flatten(start_dim=-2)
  return torch.sum(e_total * F.softmin(e_total*10.0,-1),-1)

def create_sequence(furniture,dict_cat2int,detailed=False,keep_room_dims=False, res=256):
  """
  Creates a sequence from a list of furniture objects
  """
  cat_windoor = ['window','door']
  cat_window = ['window']
  cat_door = ['door']
  is_windoor = torch.stack([torch.as_tensor(f.coarse_category in cat_windoor) for f in furniture],0)
  is_window = torch.stack([torch.as_tensor(f.coarse_category in cat_window) for f in furniture],0)
  is_door = torch.stack([torch.as_tensor(f.coarse_category in cat_door) for f in furniture],0)
  if detailed:
    val_seq = torch.stack([f.to_sequence(dict_cat2int) for f in furniture],0)
  else:
    val_seq = torch.stack([f.to_sequence(dict_cat2int) for f in furniture],0)
  pos = torch.stack([f.pos for f in furniture],0)
  dim = torch.stack([f.dim for f in furniture],0)
  front = torch.stack([f.world_front() for f in furniture],0) 
  side = torch.stack([-front[:,1].clone(),front[:,0].clone()],1) * 0.5 * dim[:,0].view(-1,1)
  front = front * 0.5 * dim[:,1].view(-1,1)
  corners = torch.stack([pos + front + side, pos + front - side, pos - front + side, pos - front - side],2)
  corners[is_windoor,:,2:4] = corners[is_windoor,:,0:2] # for doors and windows only the front is inside the room
  is_vertical = torch.logical_and(val_seq[is_door,5] % (np.pi) < (2*np.pi/res), val_seq[is_door,5] % (np.pi) > (-2*np.pi/res))
  is_horizontal = torch.logical_and(val_seq[is_door,5] % (np.pi) < (np.pi/2 + 2*np.pi/res), val_seq[is_door,5] % (np.pi) > (np.pi/2 - 2*np.pi/res))
  if torch.sum(~is_door) > 0:
    furn_corners = corners[~is_door,:] # don't use doors for room boundary
    door_corners = corners[is_door,:]
    horizontal_corners = door_corners[is_horizontal,:]
    vertical_corners = door_corners[is_vertical,:]
    room_min = torch.min(torch.min(torch.cat([furn_corners,horizontal_corners],0),2)[0],0)[0]
    room_max = torch.max(torch.max(torch.cat([furn_corners,horizontal_corners],0),2)[0],0)[0]
    room_min[0] = torch.min(torch.min(torch.cat([furn_corners[:,[0],:],vertical_corners[:,[0],:]],0),2)[0],0)[0]
    room_max[0] = torch.max(torch.max(torch.cat([furn_corners[:,[0],:],vertical_corners[:,[0],:]],0),2)[0],0)[0]
  else:
    room_min = torch.min(torch.min(corners,2)[0],0)[0]
    room_max = torch.max(torch.max(corners,2)[0],0)[0]

  if ((room_max[0]-room_min[0]) < 1e-5) or ((room_max[1]-room_min[1]) < 1e-5):
    room_min = torch.min(torch.min(corners,2)[0],0)[0]
    room_max = torch.max(torch.max(corners,2)[0],0)[0]
  
  if furniture[0].coarse_category == 'room':
    room_corners = furniture[0].bbox()
    if keep_room_dims:
      room_min = room_corners[0,:]
      room_max = room_corners[1,:]
    else:
      room_min = torch.minimum(room_min,room_corners[0,:])
      room_max = torch.maximum(room_max,room_corners[1,:])
  room_seq = torch.as_tensor([dict_cat2int['room'], 0.5*(room_max[0]+room_min[0]), 0.5*(room_max[1]+room_min[1]), (room_max[0]-room_min[0]), (room_max[1]-room_min[1]), 0.0])
  if furniture[0].coarse_category == 'room':
    return torch.cat([room_seq.view(1,-1),val_seq[1:,:]],0)
  else:
    return torch.cat([room_seq.view(1,-1),val_seq],0)

def sequence_to_furniture(sequence,dict_int2cat,device='cpu'):
  """
  Creates a list of furniture from a sequence
  """
  furniture = []
  for i in range(sequence.size(0)):
    furniture.append(Furniture(sequence=sequence[i,:],dict_int2cat=dict_int2cat,device=device))
  return furniture

def quantize_sequence(sequence,minvalue_dict,maxvalue_dict,res):
  """
  Quantizes a sequence based on the minimum and maximum values for each category
  """
  cat_ind = sequence[:,0].int().tolist()
  val_min = torch.stack([minvalue_dict[ci] for ci in cat_ind],0)
  val_max = torch.stack([maxvalue_dict[ci] for ci in cat_ind],0)
  val_min[:,0] = 0.0 # categories should not change
  val_max[:,0] = 1.0
  val_min[:,5] = -np.pi + 2 * np.pi / res # minimum rotation
  val_max[:,5] = np.pi # maximum rotation
  round_up = sequence[:,5] <= -np.pi + np.pi / res
  sequence[round_up,5] += 2 * np.pi # make sure rounding at +-pi is correct
  quantization = (sequence - val_min) / (val_max - val_min)
  quantization[:,1:] = quantization[:,1:] * (res-1)
  quantization = torch.maximum(quantization, torch.zeros_like(quantization)) # clip to range [0,res-1]
  quantization = torch.minimum(quantization, torch.ones_like(quantization) * res-1)
  return torch.round(quantization).int()

def unquantize_sequence(sequence,minvalue_dict,maxvalue_dict,res,device='cpu'):
  """
  Unquantizes a sequence based on the minimum and maximum values for each category
  """
  cat_ind = sequence[:,0].int().tolist()
  val_min = torch.stack([minvalue_dict[ci] for ci in cat_ind],0).to(device)
  val_max = torch.stack([maxvalue_dict[ci] for ci in cat_ind],0).to(device)
  val_min[:,0] = 0.0 # categories should not change
  val_max[:,0] = 1.0 # categories should not change
  val_min[:,5] = -np.pi + 2 * np.pi / res # minimum rotation
  val_max[:,5] = np.pi # maximum rotation
  res_tensor = torch.ones_like(val_min,device=device) * (res-1)
  res_tensor[:,0] = 1.0
  reconstruction = val_min + (sequence / res_tensor) * (val_max - val_min)
  return reconstruction

def quantize_sequence_grid(sequence,minvalue_dict,maxvalue_dict,res):
  """
  Quantizes a sequence based on the room dimensions (like dividing the room with a grid)
  """
  seq = sequence.clone()
  cat_ind = seq[:,0].int().tolist()

  val_min = minvalue_dict[cat_ind[0]].clone().view(1,-1)
  val_max = maxvalue_dict[cat_ind[0]].clone().view(1,-1)
  val_min[:,0:3] = 0.0 # categories should not change, position should be 0
  val_max[:,0:3] = 1.0
  val_min[:,5] = -np.pi + 2 * np.pi / res # minimum rotation
  val_max[:,5] = np.pi # maximum rotation
  val_min = val_min.repeat(sequence.size(0),1)
  val_max = val_max.repeat(sequence.size(0),1)

  is_window = (sequence[:,0].int() == 2)
  is_door = (sequence[:,0].int() == 1)
  is_windoor = torch.logical_or(is_window,is_door)
  room_quantized = (seq[0,:] - val_min[0,:]) / (val_max[0,:] - val_min[0,:])
  room_quantized[1:] = room_quantized[1:] * (res-1)
  room_quantized = torch.round(room_quantized).int()
  room_unquantized = val_min[0,:] + (room_quantized / (res-1)) * (val_max[0,:] - val_min[0,:])

  # use corner as position and move room pos to 0,0
  obj_up = torch.stack([-torch.sin(seq[:,5]),torch.cos(seq[:,5])],-1)
  obj_left = torch.stack([-obj_up[:,1],obj_up[:,0]],-1)
  seq[:,1:3] = seq[:,1:3] + obj_left[:,:] * 0.5 * seq[:,[3]] - obj_up[:,:] * 0.5 * seq[:,[4]]
  seq[:,1:3] = seq[:,1:3] - seq[[0],1:3]

  is_wider = seq[0,3] >= seq[0,4]
  if ~is_wider:
    seq[0,5] = -np.pi / 2.0
    seq[0,[3,4]] = seq[0,[4,3]]
    unit_length = room_unquantized[4] / (res-2)
    val_min[0,3] = val_min[0,4]
    val_max[0,3] = val_max[0,4]
  else:
    unit_length = room_unquantized[3] / (res-2)
  val_min[0:,1:3] = -unit_length
  val_max[0:,1:3] = unit_length * (res-2)
  val_min[1:,3] = unit_length
  val_max[1:,3] = unit_length * res
  val_min[:,4] = unit_length
  val_max[:,4] = unit_length * res

  room_min = val_min[0,:].clone()
  room_max = val_max[0,:].clone()
  room_quantized = (seq[0,:] - room_min) / (room_max - room_min)
  room_quantized[1:] = room_quantized[1:] * (res-1)
  room_quantized = torch.round(room_quantized).int()
  room_unquantized = room_min + (room_quantized / (res-1)) * (room_max - room_min)
  room_scale_diff = room_unquantized[3:5] / seq[0,3:5]

  if is_wider:
    seq[1:,[1]] = seq[1:,[1]] * room_scale_diff[0].view(-1,1)
    seq[1:,[2]] = seq[1:,[2]] * room_scale_diff[1].view(-1,1)
    seq[1:,3] = seq[1:,3] * torch.linalg.norm(room_scale_diff.view(-1,2) * obj_left[1:,:],dim=1)
    seq[1:,4] = seq[1:,4] * torch.linalg.norm(room_scale_diff.view(-1,2) * obj_up[1:,:],dim=1)
  else:
    seq[1:,[1]] = seq[1:,[1]] * room_scale_diff[1].view(-1,1)
    seq[1:,[2]] = seq[1:,[2]] * room_scale_diff[0].view(-1,1)
    seq[1:,3] = seq[1:,3] * torch.linalg.norm(room_scale_diff[[1,0]].view(-1,2) * obj_left[1:,:],dim=1)
    seq[1:,4] = seq[1:,4] * torch.linalg.norm(room_scale_diff[[1,0]].view(-1,2) * obj_up[1:,:],dim=1)
  seq[is_windoor,1:3] = seq[is_windoor,1:3] + obj_up[is_windoor,:] * 1.0 * (seq[is_windoor,4].view(-1,1) - unit_length)
  seq[is_windoor,4] = unit_length

  # quantize position and adjust length and width
  p = 1
  pos_quantized = (seq[p:,1:3] - val_min[p:,1:3]) / (val_max[p:,1:3] - val_min[p:,1:3])
  pos_quantized = pos_quantized * (res-1)
  pos_quantized = torch.round(pos_quantized).int()
  pos_unquantized = val_min[p:,1:3] + (pos_quantized / (res-1)) * (val_max[p:,1:3] - val_min[p:,1:3])
  pos_diff = pos_unquantized - seq[p:,1:3]

  round_up = seq[:,5] <= -np.pi + np.pi / res
  seq[round_up,5] += 2 * np.pi # make sure rounding at +-pi is correct
  quantization = (seq - val_min) / (val_max - val_min)
  quantization[:,1:] = quantization[:,1:] * (res-1)
  quantization = torch.round(quantization)

  # adjust doors and windows
  # angles 15, 31, 47 and 63 are possible
  windoor = quantization[is_windoor,:]
  if is_wider:
    for i in range(windoor.size(0)):
      if windoor[i,5] == (res/4-1):
        if windoor[i,1] > (res / 2 - 1):
          windoor[i,1] = res-2
        if windoor[i,1] < (res / 2 - 1):
          windoor[i,1] = 0
      if windoor[i,5] == (res/2-1):
        if windoor[i,2] > (1 + quantization[0,4] / 2):
          windoor[i,2] = quantization[0,4] + 2
        if windoor[i,2] < (1 + quantization[0,4] / 2):
          windoor[i,2] = 0
      if windoor[i,5] == (3*res/4-1):
        if windoor[i,1] > (res / 2 - 1):
          windoor[i,1] = res-1
        if windoor[i,1] < (res / 2 - 1):
          windoor[i,1] = 2
      if windoor[i,5] == (res-1):
        if windoor[i,2] > (1 + quantization[0,4] / 2):
          windoor[i,2] = quantization[0,4] + 2
        if windoor[i,2] < (1 + quantization[0,4] / 2):
          windoor[i,2] = 2
  else:
    for i in range(windoor.size(0)):
      if windoor[i,5] == (res/4-1):
        if windoor[i,1] > (1 + quantization[0,4] / 2):
          windoor[i,1] = quantization[0,4] + 1
        if windoor[i,1] < (1 + quantization[0,4] / 2):
          windoor[i,1] = 0
      if windoor[i,5] == (res/2-1):
        if windoor[i,2] > (res / 2 - 1):
          windoor[i,2] = res-2
        if windoor[i,2] < (res / 2 - 1):
          windoor[i,2] = 0
      if windoor[i,5] == (3*res/4-1):
        if windoor[i,1] > (1 + quantization[0,4] / 2):
          windoor[i,1] = quantization[0,4] + 2
        if windoor[i,1] < (1 + quantization[0,4] / 2):
          windoor[i,1] = 2
      if windoor[i,5] == (res-1):
        if windoor[i,2] > (res / 2 - 1):
          windoor[i,2] = res-1
        if windoor[i,2] < (res / 2 - 1):
          windoor[i,2] = 2
  quantization[is_windoor,:] = windoor

  quantization = torch.maximum(quantization, torch.zeros_like(quantization)) # clip to range [0,res-1]
  quantization = torch.minimum(quantization, torch.ones_like(quantization) * (res-1))

  return quantization.int()

def unquantize_sequence_grid(sequence,minvalue_dict,maxvalue_dict,res,adjust_windows=True,device='cpu'):
  """
  Unquantizes a sequence based on the room dimensions (like dividing the room with a grid)
  """
  cat_ind = sequence[:,0].int().tolist()
  is_wider = sequence[0,5] >= (res/2 - 1)
  val_min = minvalue_dict[cat_ind[0]].clone().view(1,-1).to(device)
  val_max = maxvalue_dict[cat_ind[0]].clone().view(1,-1).to(device)
  val_min[:,0] = 0.0 # categories should not change
  val_max[:,0] = 1.0
  val_min[:,5] = -np.pi + 2 * np.pi / res # minimum rotation
  val_max[:,5] = np.pi # maximum rotation
  val_min = val_min.repeat(sequence.size(0),1)
  val_max = val_max.repeat(sequence.size(0),1)

  if ~is_wider:
    val_min[0,3] = val_min[0,4]
    val_max[0,3] = val_max[0,4]
  room_width = val_min[0,3] + sequence[0,3] / (res-1) *  (val_max[0,3] - val_min[0,3])
  unit_length = room_width / (res-2)
  val_min[0:,1:3] = -unit_length
  val_max[0:,1:3] = unit_length * (res-2)
  val_min[1:,3] = unit_length
  val_max[1:,3] = unit_length * res
  val_min[:,4] = unit_length
  val_max[:,4] = unit_length * res

  # post-process doors and windows that should be outside
  if adjust_windows:
    is_window = (sequence[:,0].int() == 2)
    is_door = (sequence[:,0].int() == 1)
    is_windoor = torch.logical_or(is_window,is_door)
    windoor = sequence[is_windoor,:]
    if is_wider:
      for i in range(windoor.size(0)):
        if (windoor[i,5] == (3*res/4-1)) and (windoor[i,1] == (res-1)):
          windoor[i,1] = windoor[i,1] + 1
        if (windoor[i,5] == (res-1)) and (windoor[i,2] == 2 + sequence[0,4]):
          windoor[i,2] = windoor[i,2] + 1
    else:
      for i in range(windoor.size(0)):
        if (windoor[i,5] == (3*res/4-1)) and (windoor[i,1] == 2 + sequence[0,4]):
          windoor[i,1] = windoor[i,1] + 1
        if (windoor[i,5] == (res-1)) and (windoor[i,2] == (res-1)):
          windoor[i,2] = windoor[i,2] + 1
    sequence[is_windoor] = windoor

  res_tensor = torch.ones_like(val_min,device=device) * (res-1)
  res_tensor[:,0] = 1.0
  reconstruction = val_min + (sequence / res_tensor) * (val_max - val_min)

  if ~is_wider:
    reconstruction[0,5] = 0
    reconstruction[0,[3,4]] = reconstruction[0,[4,3]]
  obj_up = torch.cat([-torch.sin(reconstruction[:,[5]]),torch.cos(reconstruction[:,[5]])],-1)
  obj_left = torch.cat([-torch.cos(reconstruction[:,[5]]),-torch.sin(reconstruction[:,[5]])],-1)
  reconstruction[:,1:3] = reconstruction[:,1:3]  - obj_left * 0.5 * reconstruction[:,[3]] + obj_up * 0.5 * reconstruction[:,[4]]

  return reconstruction

def sequence_statistics(sequences, n_cats=20):
  """
  Computes statistics about the occurence of different furniture categories in the dataset
  """
  all_count = torch.zeros(sequences.size(0),n_cats)
  for i in range(sequences.size(0)):
    seq_cats = sequences[i,0:-2:6]
    for j in range(seq_cats.size(0)):
      if seq_cats[j] < n_cats:
        all_count[i,seq_cats[j]] += 1
  count_mean = torch.mean(all_count,0)
  count_median = torch.median(all_count,0)[0]
  count_std = torch.std(all_count,0)
  count_min = torch.min(all_count,0)[0]
  count_max = torch.max(all_count,0)[0]
  n_bins = 6
  count_hist = torch.empty(n_bins,0)
  for i in range(n_cats):
    count_hist = torch.cat([count_hist,torch.histc(all_count[:,i],bins=n_bins,min=-0.5,max=5.5).unsqueeze(1)],1)
  count_hist = torch.transpose(count_hist / sequences.size(0),0,1)
  return {"mean": count_mean,"median": count_median,"std": count_std,"min": count_min,"max": count_max,"hist": count_hist}

def plot_room(furniture, dict_cat2int, path_save_plot=None):
  """
  Creates a 2D-plot of a layout
  """
  fig = plt.figure(figsize=(10,10))
  cmap = plt.get_cmap('tab20')
  ax = fig.gca()
  xmin = 100.0
  xmax = -100.0
  ymin = 100.0
  ymax = -100.0
  for ni in range(len(furniture)):
    fobj = furniture[ni]
    cat_color = cmap(dict_cat2int[fobj.coarse_category] / (len(dict_cat2int.keys())-1))
    pos = fobj.pos
    dim = fobj.dim
    front = fobj.world_front()
    side = torch.as_tensor([-front[1].clone(),front[0].clone()]) * 0.5 * dim[0]
    front = front * 0.5 * dim[1]
    corners = torch.stack([pos - front - side, pos - front + side, pos + front + side, pos + front - side, pos - front - side],0)

    plt.plot(corners[:,0],corners[:,1],color=cat_color)
    if torch.min(corners[:,0],0)[0] < xmin:
      xmin = torch.min(corners[:,0],0)[0]
    if torch.max(corners[:,0],0)[0] > xmax:
      xmax = torch.max(corners[:,0],0)[0]
    if torch.min(corners[:,1],0)[0] < ymin:
      ymin = torch.min(corners[:,1],0)[0]
    if torch.max(corners[:,1],0)[0] > ymax:
      ymax = torch.max(corners[:,1],0)[0]
    plt.text(corners[1,0], corners[1,1], fobj.coarse_category)
    dir = fobj.world_front()
    plt.quiver(fobj.pos[0],fobj.pos[1], dir[0], dir[1], color=cat_color, scale=25, headwidth=2, headlength=3, headaxislength=3)
  plt.xlim(xmin,xmax)
  plt.ylim(ymin,ymax)
  ax.set_aspect('equal', 'datalim')
  if path_save_plot:
    plt.savefig(path_save_plot)
  else:
    plt.show()

def plot_sequences_grid(sequences, dict_int2cat, dict_cat2int, minvalue_dict, maxvalue_dict, res, text=[], path_save_plots=None):
  """
  Creates a 2D-plot of the given layout for each input sequence
  """
  n_cats = len(minvalue_dict.keys())
  room_sequences = torch.as_tensor(sequences).view(-1,127)
  for i in range(room_sequences.size(0)):
    room_seq = room_sequences[i,:-1].view(-1,6)
    valid_row = torch.logical_and(room_seq[:,0] < n_cats,torch.sum(room_seq >= res,1) < 1)
    room_seq = room_seq[valid_row,:]
    reconstructed = unquantize_sequence_grid(room_seq,minvalue_dict,maxvalue_dict,res)
    furniture = sequence_to_furniture(reconstructed,dict_int2cat)
    path_save_plot = None
    if path_save_plots:
      path_save_plot = path_save_plots + str(i) + ".png"
    plot_room(furniture,dict_cat2int,path_save_plot=path_save_plot)
    if len(text) > i:
      print(text[i])

def generate_scenes(model,n_scenes,res=256,max_seq_len=127,top_k=0,top_p=0.9,max_noncat=False,max_windoor=False):
  """
  Generates sequences using the given trained model, without collision-detection
  """
  # if max_noncat is true, pos, dim, ori of objects is not randomly sampled (except windows and doors)
  # if max_windoor is true, pos, dim, ori of windows and doors is also not randomly sampled
  with torch.no_grad():
    model.eval()
    sequence = torch.zeros(n_scenes,1,dtype=torch.long,device=model.device)
    pos_ids = torch.zeros(n_scenes,1,dtype=torch.long,device=model.device)
    ind_ids = torch.zeros(n_scenes,1,dtype=torch.long,device=model.device)
    cur_token = sequence
    past = None
    post_windoor = torch.zeros(n_scenes, dtype=torch.bool)
    for j in range(max_seq_len-1):
      i = j + 1
      input_ids = cur_token
      output = model(input_ids,position_ids=pos_ids,index_ids=ind_ids,past_key_values=past)
      next_token_logits = output.logits[:,-1, :]
      filtered_next_token_logits = transformers.top_k_top_p_filtering(next_token_logits,top_k=top_k,top_p=top_p)
      probs = F.softmax(filtered_next_token_logits, dim=-1)
      cur_token = torch.multinomial(probs, num_samples=1)
      if max_noncat and (i > 5):
        if (i % 6) == 0: 
          if max_windoor:
            post_windoor = cur_token > 0 # not room
          else:
            post_windoor = cur_token > 2 # not room, window or door
        else:
          cur_token_max = output.logits.argmax(axis=-1)
          cur_token[post_windoor] = cur_token_max[post_windoor]

      past = output.past_key_values
      sequence = torch.cat([sequence,cur_token],1)
      pos_ids += 1
      is_end = (cur_token == res).flatten()
      if res > max_seq_len:
        ind_ids[is_end,:] = torch.ones(torch.sum(is_end),1,dtype=torch.long,device=model.device) * (max_seq_len-1)
      else:
        ind_ids[is_end,:] = torch.ones(torch.sum(is_end),1,dtype=torch.long,device=model.device) * res
      ind_ids[~is_end,:] = (ind_ids[~is_end,:] + 1) % 6
    new_sequences = sequence.to('cpu')
    return new_sequences

def evaluate_scenes(sequences, minvalue_dict, maxvalue_dict, dict_cat2fun, res=256, use_log=True, grid_quantization=True, device='cpu', use_alt_loss=False):
  """
  Evaluates the layout in terms of ergonomic score for each given sequence
  """
  n_cats = len(minvalue_dict.keys())
  ergo_scores_new = []
  for i in range(sequences.size(0)):
    room_seq = sequences[i,:-1].view(-1,6)
    valid_row = torch.logical_and(room_seq[:,0] < n_cats,torch.sum(room_seq >= res,1) < 1)
    if torch.sum(valid_row) > 1:
      room_seq = room_seq[valid_row,:]
      if (room_seq[0,0] == 0):
        seq_ergo_scores = []
        seq_total_score = torch.zeros(1,device=device)
        n_losses = 0
        if grid_quantization: 
          unquantized = unquantize_sequence_grid(room_seq,minvalue_dict,maxvalue_dict,res,device=device)
        else:
          unquantized = unquantize_sequence(room_seq,minvalue_dict,maxvalue_dict,res,device=device)
        if use_alt_loss:
          seq_ergo_scores.append(score_overlap(unquantized, minvalue_dict,maxvalue_dict,res,dict_cat2fun,use_log=use_log,device=device))
        else:
          seq_ergo_scores.append(score_read_book(unquantized, minvalue_dict,maxvalue_dict,res,dict_cat2fun,use_log=use_log,device=device))
          seq_ergo_scores.append(score_watch_tv(unquantized, minvalue_dict,maxvalue_dict,res, dict_cat2fun,use_log=use_log,device=device))
          seq_ergo_scores.append(score_use_computer(unquantized, minvalue_dict,maxvalue_dict,res, dict_cat2fun,use_log=use_log,device=device))
          seq_ergo_scores.append(score_work_table(unquantized, minvalue_dict,maxvalue_dict,res, dict_cat2fun,use_log=use_log,device=device))
        for score in seq_ergo_scores:
          if score > -1.0:
            if score < 0.0:
              score = 0.0
            seq_total_score = seq_total_score + score
            n_losses = n_losses + 1
        if seq_ergo_scores[0] < -1.0:
          seq_total_score = seq_total_score + torch.ones(1,device=device)
          n_losses = n_losses + 1
          if use_log:
            seq_total_score = seq_total_score + 4.0 * torch.ones(1,device=device)
        if n_losses > 0:
          seq_total_score = seq_total_score / n_losses
        ergo_scores_new.append(seq_total_score)
      else:
        ergo_scores_new.append(torch.zeros(1,device=device))
    else:
      ergo_scores_new.append(torch.zeros(1,device=device))
  return ergo_scores_new

def generate_and_rank_scenes(n_scenes, n_versions, model_names, minvalue_dict, maxvalue_dict, dict_cat2fun, path_output_data, path_trained_models, res=256, max_seq_len=127, top_k=0, top_p=0.9, order_switch=1, max_noncat=False, max_windoor=False, max_batch_size=200, device='cpu', use_alt_loss=False):
  """
  Generates and evaluates new layouts, then sorts them based on ergonomic score and saves them
  """
  # order switch: 0 - no switch; 1 - orient,dim,pos; 2 - orient,pos,dim 
  # max_noncat: if True, pos, dim and loc are not sampled, but the best option is chosen (except windows and doors)
  # if max_windoor is true, pos, dim, ori of windows and doors is also not randomly sampled
  n_batches = 1
  if n_scenes > max_batch_size:
    n_batches = int(n_scenes / max_batch_size)
    n_scenes = max_batch_size
  if top_k > 0:
    sampling_type = "k" + str(top_k)
  else:
    sampling_type = "p" + str(top_p).replace(".", "")
  if max_noncat:
    sampling_type = sampling_type + "max"
    if max_windoor:
      sampling_type = sampling_type + "wd"

  model_mean_scores = []
  model_median_scores = []
  model_variance_scores = []
  for i in range(len(model_names)):
    model_name = 'model_' + model_names[i]
    version_mean_scores = []
    for version in range(n_versions):
      print("Evaluating", model_name, "- Epoch", version+1, "of", n_versions, end="\r")
      version_name = "model_tmp" + str(version)
      model = GPT2LMHeadModelCustom.from_pretrained(path_trained_models + model_name + "/" + version_name + "/")
      model.to(device)

      ergo_scores = []
      all_sequences = torch.empty(0,max_seq_len,dtype=torch.long)
      for i in range(n_batches):
        sequences = generate_scenes(model,n_scenes,res=res,max_seq_len=max_seq_len,top_k=top_k,top_p=top_p,max_noncat=max_noncat)
        if order_switch > 0:
          last_tokens = sequences[:,-1]
          sequences = sequences[:,:-1].view(sequences.size(0),-1,6)
          if order_switch == 1:
            sequences = sequences[:,:,[0,4,5,2,3,1]].view(sequences.size(0),-1)
          else:
            sequences = sequences[:,:,[0,2,3,4,5,1]].view(sequences.size(0),-1)
          sequences = torch.cat([sequences,last_tokens.view(-1,1)],1)
        all_sequences = torch.cat([all_sequences,sequences],0)
        ergo_scores.extend(evaluate_scenes(sequences, minvalue_dict, maxvalue_dict, dict_cat2fun, res=res, use_log=True,grid_quantization=True, device='cpu',
          use_alt_loss=use_alt_loss))

      ergo_scores = torch.cat(ergo_scores,0)
      scores_sorted, indices_sorted = torch.sort(ergo_scores)
      version_mean_scores.append(torch.mean(ergo_scores).item())
      if not os.path.isdir(path_output_data + model_name):
        os.mkdir(path_output_data + model_name)
      if not os.path.isdir(path_output_data + model_name + "/" + version_name):
        os.mkdir(path_output_data + model_name + "/" + version_name)
      pickle.dump(scores_sorted, open(path_output_data + model_name + "/" + version_name + "/" + sampling_type + "_scores.pkl", "wb" ))
      pickle.dump(all_sequences[indices_sorted,:], open(path_output_data + model_name + "/" + version_name + "/" + sampling_type + "_sequences.pkl", "wb" ))
      
    print()
    model_mean_scores.append(version_mean_scores)
    pickle.dump(version_mean_scores, open(path_output_data + model_name + "/" + sampling_type + "_mean_scores.pkl", "wb" ))
  return sampling_type

def signed_dist(p, dim, center=torch.zeros(1,2), theta=torch.zeros(1,),invert=False):
  """
  Computes the signed distance from each point p to the bounding box given by dim, center and theta
  """
  n_p = p.size(0)
  device = p.device
  R = torch.as_tensor([[torch.cos(theta),-torch.sin(theta)],[torch.sin(theta),torch.cos(theta)]],device=theta.device)
  pt = torch.matmul(p,R) - torch.matmul(center,R)
  d = torch.abs(pt)-0.5*dim.view(1,2)
  if invert:
    return torch.maximum(torch.zeros(n_p,device=device),torch.linalg.norm(torch.maximum(d,torch.zeros(n_p,2,device=device)),dim=1) + torch.minimum(torch.maximum(d[:,0],d[:,1]),torch.zeros(n_p,device=device)))
  else:
    return -torch.minimum(torch.zeros(n_p,),torch.linalg.norm(torch.maximum(d,torch.zeros(n_p,2)),dim=1) + torch.minimum(torch.maximum(d[:,0],d[:,1]),torch.zeros(n_p)))

def signed_dist_par(p, dim, center=torch.zeros(1,2), theta=torch.zeros(1,),invert=False):
  """
  Computes the signed distance from each point p to each bounding box given by dim, center and theta in parallel
  """
  n_p = p.size(0)
  device = p.device
  R = torch.stack([torch.stack([torch.cos(theta),-torch.sin(theta)],dim=-1),torch.stack([torch.sin(theta),torch.cos(theta)],dim=-1)],dim=-1)
  pt = (torch.matmul(p.view(n_p,1,1,2),R) - torch.matmul(center.view(n_p,-1,1,2),R)).squeeze(dim=2)
  d = torch.abs(pt)-0.5*dim
  if invert:
    return torch.maximum(torch.zeros(d.size(0),d.size(1),device=device),torch.linalg.norm(torch.maximum(d,torch.zeros_like(d,device=device)),dim=-1) 
                         + torch.minimum(torch.maximum(d[:,:,0],d[:,:,1]),torch.zeros(d.size(0),d.size(1),device=device)))
  else:
    return -torch.minimum(torch.zeros(d.size(0),d.size(1),device=device),torch.linalg.norm(torch.maximum(d,torch.zeros_like(d,device=device)),dim=-1) 
                          + torch.minimum(torch.maximum(d[:,:,0],d[:,:,1]),torch.zeros(d.size(0),d.size(1),device=device)))

def eval_overlap(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun,use_log=True,device='cpu'):
  """
  Evaluates the Overlap loss for each predicted token value
  """
  ind_posdimrot = [1,2,3,4,5]
  cat_all = torch.linspace(1,len(minvalue_dict.keys())-1,len(minvalue_dict.keys())-1).view(1,-1).to(device)
  is_furn = torch.sum(torch.eq(cat_all,label_mat[:,[0]]),-1) > 0
  is_valid = torch.sum(interp_val > (res-0.9),1) + torch.sum(label_mat > (res-0.9),1) < 1.0
  pos_room = label_mat[0,1:3]
  dim_room = label_mat[0,3:5]
  rot_room = label_mat[0,5].view(1,)
  interp_val = interp_val[is_furn,:]
  label_mat = label_mat[is_furn,:]

  if torch.sum(cat_all) < 1:
    return -2.0*torch.ones(1,).to(device)
  categories = interp_val[:,0].view(-1,1)
  interp_val = interp_val[:,ind_posdimrot]
  value_mat = label_mat[:,ind_posdimrot].clone()

  # replace gt values with predicted values
  n_rows = interp_val.size(0)
  n_cols = interp_val.size(1)
  n_elem = n_rows * n_cols
  value_mat = value_mat.repeat(n_elem,1,1)
  elem_ind = torch.linspace(0,n_elem-1,n_elem,dtype=torch.long,device=device)
  col_ind = torch.linspace(0,n_cols-1,n_cols,dtype=torch.long,device=device).repeat_interleave(n_rows)
  row_ind = torch.linspace(0,n_rows-1,n_rows,dtype=torch.long,device=device).repeat(n_cols)
  value_mat[elem_ind,row_ind,col_ind] = interp_val[row_ind,col_ind]

  # check which categories do not collide
  is_supporter = torch.sum(torch.eq(dict_cat2fun['supporter'],categories),-1) > 0
  is_supported = torch.sum(torch.eq(dict_cat2fun['supported'],categories),-1) > 0
  is_chair = torch.sum(torch.eq(dict_cat2fun['chair'],categories),-1) > 0
  is_table = torch.sum(torch.eq(dict_cat2fun['table'],categories),-1) > 0
  is_near_ceil = torch.sum(torch.eq(dict_cat2fun['near_ceil'],categories),-1) > 0
  is_windoor = torch.sum(torch.eq(dict_cat2fun['windoor'],categories),-1) > 0
  mat_supp = (is_supporter.view(-1,1) * is_supported.view(1,-1)) + (is_supported.view(-1,1) * is_supporter.view(1,-1)) == 0
  mat_chairtable = (is_chair.view(-1,1) * is_table.view(1,-1)) + (is_table.view(-1,1) * is_chair.view(1,-1)) == 0
  mat_ceil = (is_near_ceil.view(-1,1) * is_near_ceil.view(1,-1)) + (~is_near_ceil.view(-1,1) * ~is_near_ceil.view(1,-1))
  coll_check_mat = mat_supp * mat_chairtable * mat_ceil
  inv_eye = ~torch.eye(coll_check_mat.size(0),coll_check_mat.size(0),dtype=torch.bool)
  coll_check_mat = ~coll_check_mat[inv_eye].view(n_rows,n_rows-1)

  ind_mat = torch.zeros_like(value_mat,dtype=torch.bool)
  ind_mat[elem_ind,row_ind,:] = 1
  matA = value_mat[ind_mat].view(n_elem,1,5)
  matB = value_mat[~ind_mat].view(n_elem,-1,5)

  all_pos = matA[:,:,0:2]
  all_dim = matA[:,:,2:4]
  all_rot = matA[:,:,4].view(n_elem,1,1)
  all_front = torch.cat([-torch.sin(all_rot),torch.cos(all_rot)],dim=-1) * 0.5 * all_dim[:,:,[1]]
  all_side = torch.cat([torch.cos(all_rot),torch.sin(all_rot)],dim=-1) * 0.5 * all_dim[:,:,[0]]
  all_samples = torch.concat([all_pos, all_pos + all_front, all_pos - all_side, all_pos - all_front, all_pos + all_side,
                             all_pos + all_front + all_side, all_pos + all_front - all_side, all_pos - all_front - all_side, all_pos - all_front + all_side],dim=0)

  node_weights = torch.as_tensor([16.0,4.0,4.0,4.0,4.0,1.0,1.0,1.0,1.0],device=device).repeat_interleave(n_elem)
  weights_sum = torch.sum(torch.as_tensor([16.0,4.0,4.0,4.0,4.0,1.0,1.0,1.0,1.0],device=device))
  e_room = signed_dist(all_samples.squeeze(), dim_room, center=pos_room, theta=rot_room,invert=True)
  e_objects = signed_dist_par(all_samples, matB[:,:,2:4].repeat(9,1,1), center=matB[:,:,0:2].repeat(9,1,1), theta=matB[:,:,4].repeat(9,1),invert=False)
  e_room[is_windoor.repeat(9*5)] = 0.0
  e_objects[coll_check_mat.repeat(9*5,1)] = 0.0
  e_total = torch.sum(e_objects * node_weights.view(-1,1))
  e_total = e_total + torch.sum(e_room * node_weights)
  return e_total / weights_sum * 5.0 #9.0

def score_overlap(labels, minvalue_dict,maxvalue_dict,res,dict_cat2fun,use_log=True,device='cpu'):
  """
  Evaluates the Overlap loss for the given layout
  """
  label_mat = labels.clone()
  ind_posdimrot = [1,2,3,4,5]
  cat_all = torch.linspace(1,len(minvalue_dict.keys())-1,len(minvalue_dict.keys())-1).view(1,-1).to(device)
  is_furn = torch.sum(torch.eq(cat_all,label_mat[:,[0]]),-1) > 0
  is_valid = torch.sum(label_mat > (res-0.9),1) < 1.0
  pos_room = label_mat[0,1:3]
  dim_room = label_mat[0,3:5]
  rot_room = label_mat[0,5].view(1,)
  label_mat = label_mat[torch.logical_and(is_furn, is_valid),:]

  if torch.sum(cat_all) < 1:
    return -2.0*torch.ones(1,).to(device)

  # check which categories do not collide
  n_elem = label_mat.size(0)
  categories = label_mat[:,[0]].view(-1,1)
  is_supporter = torch.sum(torch.eq(dict_cat2fun['supporter'],categories),-1) > 0
  is_supported = torch.sum(torch.eq(dict_cat2fun['supported'],categories),-1) > 0
  is_chair = torch.sum(torch.eq(dict_cat2fun['chair'],categories),-1) > 0
  is_table = torch.sum(torch.eq(dict_cat2fun['table'],categories),-1) > 0
  is_near_ceil = torch.sum(torch.eq(dict_cat2fun['near_ceil'],categories),-1) > 0
  is_windoor = torch.sum(torch.eq(dict_cat2fun['windoor'],categories),-1) > 0
  mat_supp = (is_supporter.view(-1,1) * is_supported.view(1,-1)) + (is_supported.view(-1,1) * is_supporter.view(1,-1)) == 0
  mat_chairtable = (is_chair.view(-1,1) * is_table.view(1,-1)) + (is_table.view(-1,1) * is_chair.view(1,-1)) == 0
  mat_ceil = (is_near_ceil.view(-1,1) * is_near_ceil.view(1,-1)) + (~is_near_ceil.view(-1,1) * ~is_near_ceil.view(1,-1))
  coll_check_mat = mat_supp * mat_chairtable * mat_ceil
  inv_eye = ~torch.eye(coll_check_mat.size(0),coll_check_mat.size(0),dtype=torch.bool)
  coll_check_mat = ~coll_check_mat[inv_eye].view(n_elem,n_elem-1)

  elem_ind = torch.linspace(0,n_elem-1,n_elem,dtype=torch.long,device=device)
  label_mat = label_mat.repeat(n_elem,1,1)
  ind_mat = torch.zeros_like(label_mat,dtype=torch.bool)
  ind_mat[elem_ind,elem_ind,:] = 1
  matA = label_mat[ind_mat].view(n_elem,1,6)
  matB = label_mat[~ind_mat].view(n_elem,-1,6)

  all_pos = matA[:,:,1:3]
  all_dim = matA[:,:,3:5]
  all_rot = matA[:,:,5].view(n_elem,1,1)
  all_front = torch.cat([-torch.sin(all_rot),torch.cos(all_rot)],dim=-1) * 0.5 * all_dim[:,:,[1]]
  all_side = torch.cat([torch.cos(all_rot),torch.sin(all_rot)],dim=-1) * 0.5 * all_dim[:,:,[0]]
  all_samples = torch.concat([all_pos, all_pos + all_front, all_pos - all_side, all_pos - all_front, all_pos + all_side,
                             all_pos + all_front + all_side, all_pos + all_front - all_side, all_pos - all_front - all_side, all_pos - all_front + all_side],dim=0)

  node_weights = torch.as_tensor([16.0,4.0,4.0,4.0,4.0,1.0,1.0,1.0,1.0],device=device).repeat_interleave(n_elem)
  weights_sum = torch.sum(torch.as_tensor([16.0,4.0,4.0,4.0,4.0,1.0,1.0,1.0,1.0],device=device))
  e_room = signed_dist(all_samples.squeeze(), dim_room, center=pos_room, theta=rot_room,invert=True)
  e_objects = signed_dist_par(all_samples, matB[:,:,3:5].repeat(9,1,1), center=matB[:,:,1:3].repeat(9,1,1), theta=matB[:,:,5].repeat(9,1),invert=False)
  e_room[is_windoor.repeat(9)] = 0.0
  e_objects[coll_check_mat.repeat(9,1)] = 0.0
  e_total = torch.sum(e_objects * node_weights.view(-1,1))
  e_total = e_total + torch.sum(e_room * node_weights)
  return e_total / weights_sum * 5.0

def evaluate_scenes_overlap(sequences, minvalue_dict, maxvalue_dict, dict_cat2fun, use_log=True, grid_quantization=True, device='cpu'):
  """
  Evaluates the layout in terms of Overlap loss for each given sequence
  """
  n_cats = len(minvalue_dict.keys())
  ergo_scores_new = []
  for i in range(sequences.size(0)):
    room_seq = sequences[i,:-1].view(-1,6)
    valid_row = torch.logical_and(room_seq[:,0] < n_cats,torch.sum(room_seq >= res,1) < 1)
    if torch.sum(valid_row) > 1:
      room_seq = room_seq[valid_row,:]
      if (room_seq[0,0] == 0):
        seq_ergo_scores = []
        seq_total_score = torch.zeros(1,device=device)
        n_losses = 0
        if grid_quantization: 
          unquantized = unquantize_sequence_grid(room_seq,minvalue_dict,maxvalue_dict,res,device=device)
        else:
          unquantized = unquantize_sequence(room_seq,minvalue_dict,maxvalue_dict,res,device=device)
        seq_ergo_scores.append(score_overlap(unquantized, minvalue_dict,maxvalue_dict,res,dict_cat2fun,use_log=use_log,device=device))
        for score in seq_ergo_scores:
          if score > -1.0:
            if score < 0.0:
              score = 0.0
            seq_total_score = seq_total_score + score
            n_losses = n_losses + 1
        if seq_ergo_scores[0] < -1.0:
          seq_total_score = seq_total_score + torch.ones(1,device=device)
          n_losses = n_losses + 1
          if use_log:
            seq_total_score = seq_total_score + 4.0 * torch.ones(1,device=device)
        if n_losses > 0:
          seq_total_score = seq_total_score / n_losses
        ergo_scores_new.append(seq_total_score)
      else:
        ergo_scores_new.append(torch.zeros(1,device=device))
    else:
      ergo_scores_new.append(torch.zeros(1,device=device))
  return ergo_scores_new
  
def get_intersection_area(bbox_0, bbox_1): # taken from sceneformer
  """
  Computes the intersection area between two (axis-aligned) bounding boxes (y-coordinate is the 3rd value)
  """
  x_min = max(bbox_0[0, 0].item(), bbox_1[0, 0].item())
  y_min = max(bbox_0[0, 2].item(), bbox_1[0, 2].item())

  x_max = min(bbox_0[1, 0].item(), bbox_1[1, 0].item())
  y_max = min(bbox_0[1, 2].item(), bbox_1[1, 2].item())

  interArea = abs(max((x_max - x_min, 0)) * max((y_max - y_min), 0))

  return interArea

def get_intersection_area2d(bbox_0, bbox_1): # taken from sceneformer
  """
  Computes the intersection area between two (axis-aligned) bounding boxes (y-coordinate is the 2nd value)
  """
  x_min = max(bbox_0[0, 0].item(), bbox_1[0, 0].item())
  y_min = max(bbox_0[0, 1].item(), bbox_1[0, 1].item())

  x_max = min(bbox_0[1, 0].item(), bbox_1[1, 0].item())
  y_max = min(bbox_0[1, 1].item(), bbox_1[1, 1].item())

  interArea = abs(max((x_max - x_min, 0)) * max((y_max - y_min), 0))

  return interArea