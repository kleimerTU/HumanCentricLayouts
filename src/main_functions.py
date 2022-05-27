import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
from torch.nn import functional as F
import transformers

from src.modeling_gpt2_custom import GPT2LMHeadModelCustom
#from src.evaluation import *

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
    Returns the bottom left and top right axis-aligned bounding box corners of the furniture object
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

def get_predictions(labels, logits,minvalue_dict,maxvalue_dict,res,sigma=8.0,sample_random=False,device='cpu'):
  """
  Computes the interpolated value of a predicted token based on its neighborhood
  """
  logits = F.softmax(logits,-1)
  if sample_random:
    pred_ind = torch.multinomial(logits, num_samples=1)
  else:
    pred_ind = torch.argmax(logits,-1)
  gauss_weights = gaussian1d(pred_ind,sigma,res,device=device)
  row_ind = torch.linspace(0,logits.size(0)-1,logits.size(0),dtype=torch.long,device=device)
  col_ind = torch.linspace(0,logits.size(1)-2,logits.size(1)-1,dtype=torch.long,device=device)
  interp_val = torch.sum(col_ind * logits[row_ind,:-1] * gauss_weights,1)
  prob_sum = torch.sum(logits[row_ind,:-1] * gauss_weights,1)
  interp_val[prob_sum > 0.0] = interp_val[prob_sum > 0.0] / prob_sum[prob_sum > 0.0]
  interp_val = torch.cat([torch.zeros(1,device=device),interp_val[:-2]]).view(-1,6)
  label_mat = labels[:-1].view(-1,6).clone()
  return interp_val, label_mat


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
    furn_corners = corners[~is_door,:,:] # don't use doors for room boundary
    door_corners = corners[is_door,:,:]
    horizontal_corners = door_corners[is_horizontal,:,:]
    vertical_corners = door_corners[is_vertical,:,:]
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
    
def adjust_windoors(room_bb, furniture):
  """
  Adjusts the position of windows and doors so they stick to the room boundary
  """
  room_front = torch.as_tensor([0.0,1.0])
  for furn in furniture:
    if furn.coarse_category in ["door","window"]:
      is_horizontal = torch.abs(torch.dot(room_front,furn.world_front())) > 0.9
      is_vertical = torch.abs(torch.dot(room_front,furn.world_front())) < 0.1
      if is_horizontal:
        furn_borders = furn.pos + furn.world_front() * furn.dim[1]/2
        border_dists = room_bb[:,[1]] - furn_borders[1].view(1,1)
        min_index = torch.argmin(torch.abs(border_dists))
        furn.pos[1] = furn.pos[1] + border_dists.flatten()[min_index]
      elif is_vertical:
        furn_borders = furn.pos + furn.world_front() * furn.dim[1]/2
        border_dists = room_bb[:,[0]] - furn_borders[0].view(1,1)
        min_index = torch.argmin(torch.abs(border_dists))
        furn.pos[0] = furn.pos[0] + border_dists.flatten()[min_index]
  return furniture

def quantize_sequence(sequence,minvalue_dict,maxvalue_dict,res,corner_pos=False):
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
  
  if corner_pos:
    front = torch.stack([-torch.sin(sequence[:,5]),torch.cos(sequence[:,5])],-1)
    side = torch.stack([-front[:,1],front[:,0]],-1)
    sequence[:,1:3] = sequence[:,1:3] + side[:,:] * 0.5 * sequence[:,[3]] - front[:,:] * 0.5 * sequence[:,[4]]

  quantization = (sequence - val_min) / (val_max - val_min)
  quantization[:,1:] = quantization[:,1:] * (res-1)
  quantization = torch.maximum(quantization, torch.zeros_like(quantization)) # clip to range [0,res-1]
  quantization = torch.minimum(quantization, torch.ones_like(quantization) * res-1)
  return torch.round(quantization).int()

def unquantize_sequence(sequence,minvalue_dict,maxvalue_dict,res,corner_pos=False,device='cpu'):
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
  
  if corner_pos:
    front = torch.stack([-torch.sin(reconstruction[:,5]),torch.cos(reconstruction[:,5])],-1)
    side = torch.stack([-front[:,1],front[:,0]],-1)
    reconstruction[:,1:3] = reconstruction[:,1:3] - side[:,:] * 0.5 * reconstruction[:,[3]] + front[:,:] * 0.5 * reconstruction[:,[4]]

  return reconstruction

def sequence_statistics(sequences, n_cats=31, res=256):
  """
  Computes statistics about the occurence of different furniture categories in the dataset
  """
  all_count = torch.zeros(sequences.size(0),n_cats)
  attr_count = torch.zeros(n_cats,5,res)
  room_sizes = torch.zeros(sequences.size(0),2)
  for i in range(sequences.size(0)):
    room_sizes[i,:] = sequences[i,3:5]
    seq_cats = sequences[i,0:-2:6]
    for j in range(seq_cats.size(0)):
      if seq_cats[j] < n_cats:
        cat_ind = int(seq_cats[j])
        all_count[i,cat_ind] += 1
        for attr_ind in range(5):
          attr_val = sequences[i,j*6+attr_ind+1].long()
          if attr_val < res:
            attr_count[cat_ind,attr_ind,attr_val] += 1
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
  statistics = {}
  statistics["mean"] = count_mean
  statistics["median"] = count_median
  statistics["std"] = count_std
  statistics["min"] = count_min
  statistics["max"] = count_max
  statistics["hist"] = count_hist
  statistics["per_room_count"] = all_count
  statistics["room_sizes"] = room_sizes
  statistics["attribute_count"] = attr_count
  return statistics

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
    
def plot_sequences(sequences, dict_int2cat, dict_cat2int, minvalue_dict, maxvalue_dict, res, text=[], path_save_plots=None):
  """
  Creates a 2D-plot of the given layout for each input sequence
  """
  n_cats = len(minvalue_dict.keys())
  room_sequences = torch.as_tensor(sequences).view(-1,127)
  for i in range(room_sequences.size(0)):
    room_seq = room_sequences[i,:-1].view(-1,6)
    valid_row = torch.logical_and(room_seq[:,0] < n_cats,torch.sum(room_seq >= res,1) < 1)
    room_seq = room_seq[valid_row,:]
    reconstructed = unquantize_sequence(room_seq,minvalue_dict,maxvalue_dict,res)
    furniture = sequence_to_furniture(reconstructed,dict_int2cat)
    path_save_plot = None
    if path_save_plots:
      path_save_plot = path_save_plots + str(i) + ".png"
    plot_room(furniture,dict_cat2int,path_save_plot=path_save_plot)
    if len(text) > i:
      print(text[i])

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