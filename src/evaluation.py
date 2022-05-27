import torch
from torch.nn import functional as F

from src.main_functions import *

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
  E = ((torch.sum(vdotlp,-1)+1.0)/2.0)**1
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
  E = (1.0-E)**1
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
  E = 1.0-((torch.sum(vdotpg,-1)+1.0)/2.0)**1
  return E

def reach_par(pos,grid,easy_reach=0.8,dropoff=15.0):
  """
  Evaluates the reach ergonomic score
  """
  pg = torch.unsqueeze(grid,1)-torch.unsqueeze(pos,2)
  pg = torch.linalg.norm(pg,dim=-1,keepdim=False)
  E = 1.0 / (1.0 + torch.exp(-dropoff * (pg - easy_reach)))
  return E
  
def aabb_iou_par(boxesA, boxesB, invert=False): # taken from sceneformer
  """
  Computes the intersection area between pairs of (axis-aligned) bounding boxes

  boxesA: tensor with shape (n_boxesA, 1, 2, 2)
  boxesB: tensor with shape (n_boxesA, n_boxesB, 2, 2)
  invert: if true, boxesB should be inside boxesA
  out:    tensor with shape (n_boxesA, n_boxesB) (intersection area of every box in A with every box in the same row of B)
  """
  margin = 0.0
  x_min = torch.maximum(boxesA[:,:,0, 0]-margin, boxesB[:,:,0, 0]-margin)
  y_min = torch.maximum(boxesA[:,:,0, 1]-margin, boxesB[:,:,0, 1]-margin)

  x_max = torch.minimum(boxesA[:,:,1, 0]+margin, boxesB[:,:,1, 0]+margin)
  y_max = torch.minimum(boxesA[:,:,1, 1]+margin, boxesB[:,:,1, 1]+margin)

  boxesA_area = (boxesA[:,:,1,0] - boxesA[:,:,0, 0]) * (boxesA[:,:,1,1] - boxesA[:,:,0,1])
  boxesB_area = (boxesB[:,:,1,0] - boxesB[:,:,0, 0]) * (boxesB[:,:,1,1] - boxesB[:,:,0,1])
  inter_area = torch.abs(torch.maximum(x_max - x_min, torch.zeros_like(x_max)) * torch.maximum(y_max - y_min, torch.zeros_like(y_max)))
  union_area = boxesA_area #+ boxesB_area - inter_area
  if invert:
    outside_area = (boxesB_area - inter_area) / boxesB_area
    return outside_area
  else:
    return inter_area / union_area

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
  
def score_access(labels, minvalue_dict,maxvalue_dict,res,dict_cat2fun,use_log=True,device='cpu'):
  """
  Evaluates the Accessibility loss for the given layout
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
  is_sit = torch.sum(torch.eq(dict_cat2fun['sit'],categories),-1) > 0
  is_bed = torch.sum(torch.eq(dict_cat2fun['bed'],categories),-1) > 0
  is_table = torch.sum(torch.eq(dict_cat2fun['table'],categories),-1) > 0
  is_near_ceil = torch.sum(torch.eq(dict_cat2fun['near_ceil'],categories),-1) > 0
  is_tall = torch.sum(torch.eq(dict_cat2fun['tall'],categories),-1) > 0
  is_door = torch.sum(torch.eq(dict_cat2fun['door'],categories),-1) > 0
  is_window = torch.sum(torch.eq(dict_cat2fun['window'],categories),-1) > 0
  is_windoor = torch.sum(torch.eq(dict_cat2fun['windoor'],categories),-1) > 0
  mat_supp = (is_supporter.view(-1,1) * is_supported.view(1,-1)) + (is_supported.view(-1,1) * is_supporter.view(1,-1)) == 0
  mat_sittable = (is_sit.view(-1,1) * is_table.view(1,-1)) + (is_table.view(-1,1) * is_sit.view(1,-1)) == 0
  mat_wintall = (is_window.view(-1,1) * is_tall.view(1,-1)) + (~is_window.view(-1,1) * ~is_window.view(1,-1))
  mat_ceil = ~is_near_ceil.view(-1,1) * ~is_near_ceil.view(1,-1)
  coll_mat = mat_supp * mat_sittable * mat_ceil * mat_wintall
  inv_eye = ~torch.eye(coll_mat.size(0),coll_mat.size(0),dtype=torch.bool)
  coll_mat = ~coll_mat[inv_eye].view(n_elem,n_elem-1)
  coll_room = ((is_windoor + is_near_ceil) > 0).view(-1,1)
  coll_mat = torch.cat([coll_room, coll_mat], dim=1)

  elem_ind = torch.linspace(0,n_elem-1,n_elem,dtype=torch.long,device=device)
  label_mat = label_mat.repeat(n_elem,1,1)
  ind_mat = torch.zeros_like(label_mat,dtype=torch.bool)
  ind_mat[elem_ind,elem_ind,:] = 1
  matA = label_mat[ind_mat].view(n_elem,1,6)
  matB = label_mat[~ind_mat].view(n_elem,-1,6)

  is_default = ~torch.logical_or(torch.logical_or(is_door,is_window),is_bed)
  is_default = is_default
  all_pos = matA[is_default,:,1:3]
  all_dim = matA[is_default,:,3:5]
  all_rot = matA[is_default,:,5].view(-1,1,1)
  all_front_dir = torch.cat([-torch.sin(all_rot),torch.cos(all_rot)],dim=-1)
  all_front = torch.cat([-torch.sin(all_rot),torch.cos(all_rot)],dim=-1) * 0.5 * all_dim[:,:,[1]]
  all_side = torch.cat([torch.cos(all_rot),torch.sin(all_rot)],dim=-1) * 0.5 * all_dim[:,:,[0]]
  is_bed = is_bed
  bed_pos = matA[is_bed,:,1:3]
  bed_dim = matA[is_bed,:,3:5]
  bed_rot = matA[is_bed,:,5].view(-1,1,1)
  bed_front = torch.cat([-torch.sin(bed_rot),torch.cos(bed_rot)],dim=-1) * 0.5 * bed_dim[:,:,[1]]
  bed_side_dir = torch.cat([torch.cos(bed_rot),torch.sin(bed_rot)],dim=-1)
  bed_side = bed_side_dir * 0.5 * bed_dim[:,:,[0]]
  is_windoor = is_windoor
  windoor_pos = matA[is_windoor,:,1:3]
  windoor_dim = matA[is_windoor,:,3:5]
  windoor_rot = matA[is_windoor,:,5].view(-1,1,1)
  windoor_front_dir = torch.cat([-torch.sin(windoor_rot),torch.cos(windoor_rot)],dim=-1)
  windoor_front = windoor_front_dir * 0.5 * windoor_dim[:,:,[1]]
  windoor_side = torch.cat([torch.cos(windoor_rot),torch.sin(windoor_rot)],dim=-1) * 0.5 * windoor_dim[:,:,[0]]

  area = 0.5
  default_cornersA = torch.stack([all_pos + all_front + all_side, all_pos + all_front + all_side + area * all_front_dir,
                                      all_pos + all_front - all_side, all_pos + all_front - all_side + area * all_front_dir],dim=0)
  bed_corners1 = torch.stack([bed_pos + 0.5 * bed_front + bed_side, bed_pos + 0.5 * bed_front + bed_side + area * bed_side_dir,
                              bed_pos - 0.5 * bed_front + bed_side, bed_pos - 0.5 * bed_front + bed_side + area * bed_side_dir],dim=0)
  bed_corners2 = torch.stack([bed_pos + 0.5 * bed_front - bed_side, bed_pos + 0.5 * bed_front - bed_side - area * bed_side_dir,
                              bed_pos - 0.5 * bed_front - bed_side, bed_pos - 0.5 * bed_front - bed_side - area * bed_side_dir],dim=0)
  windoor_corners1 = torch.stack([windoor_pos + windoor_front + windoor_side, windoor_pos + windoor_front + windoor_side + area * windoor_front_dir,
                                  windoor_pos + windoor_front - windoor_side, windoor_pos + windoor_front - windoor_side + area * windoor_front_dir],dim=0)
  windoor_corners2 = torch.stack([windoor_pos - windoor_front + windoor_side, windoor_pos - windoor_front + windoor_side - area * windoor_front_dir,
                                  windoor_pos - windoor_front - windoor_side, windoor_pos - windoor_front - windoor_side - area * windoor_front_dir],dim=0)

  bboxes_default = torch.stack([torch.min(default_cornersA,0)[0],torch.max(default_cornersA,0)[0]],dim=2)
  bbboxes_bed1 = torch.stack([torch.min(bed_corners1,0)[0],torch.max(bed_corners1,0)[0]],dim=2)
  bbboxes_bed2 = torch.stack([torch.min(bed_corners2,0)[0],torch.max(bed_corners2,0)[0]],dim=2)
  bbboxes_windoor1 = torch.stack([torch.min(windoor_corners1,0)[0],torch.max(windoor_corners1,0)[0]],dim=2)
  bbboxes_windoor2 = torch.stack([torch.min(windoor_corners2,0)[0],torch.max(windoor_corners2,0)[0]],dim=2)
  bboxesA = torch.cat([bboxes_default,bbboxes_bed1,bbboxes_bed2,bbboxes_windoor1,bbboxes_windoor2],dim=0)
  coll_mat_full = torch.cat([coll_mat[is_default,:],coll_mat[is_bed,:],coll_mat[is_bed,:],coll_mat[is_windoor,:],coll_mat[is_windoor,:]],dim=0)
  range_def = [0,bboxes_default.size(0)]
  range_bed1 = [range_def[1],range_def[1]+bbboxes_bed1.size(0)]
  range_bed2 = [range_bed1[1],range_bed1[1]+bbboxes_bed2.size(0)]
  range_windoor1 = [range_bed2[1],range_bed2[1]+bbboxes_windoor1.size(0)]
  range_windoor2 = [range_windoor1[1],range_windoor1[1]+bbboxes_windoor2.size(0)]

  all_posB = matB[:,:,1:3]
  all_dimB = matB[:,:,3:5]
  all_rotB = matB[:,:,5].view(n_elem,-1,1)
  all_frontB = torch.cat([-torch.sin(all_rotB),torch.cos(all_rotB)],dim=-1) * 0.5 * all_dimB[:,:,[1]]
  all_sideB = torch.cat([torch.cos(all_rotB),torch.sin(all_rotB)],dim=-1) * 0.5 * all_dimB[:,:,[0]]
  all_cornersB = torch.stack([all_posB + all_frontB + all_sideB, all_posB + all_frontB - all_sideB, all_posB - all_frontB - all_sideB, all_posB - all_frontB + all_sideB],dim=0)
  min_cornersB = torch.min(all_cornersB,0)[0]
  max_cornersB = torch.max(all_cornersB,0)[0]
  bboxesB = torch.stack([min_cornersB,max_cornersB],dim=2)
  bboxesB = torch.cat([bboxesB[is_default,:,:,:],bboxesB[is_bed,:,:,:],bboxesB[is_bed,:,:,:],bboxesB[is_windoor,:,:,:],bboxesB[is_windoor,:,:,:]],dim=0)

  bbox_room = torch.stack([pos_room-dim_room/2,pos_room+dim_room/2],dim=0).view(1,1,2,2)
  e_room = aabb_iou_par(bbox_room, bboxesA.permute(1,0,2,3),invert=True).view(-1,1)
  e_objects = aabb_iou_par(bboxesA, bboxesB)

  e_total = torch.cat([e_room,e_objects],dim=1)
  e_total[coll_mat_full] = torch.zeros(1,)
  e_total = torch.sum(e_total,dim=1)

  e_total_default = e_total[range_def[0]:range_def[1]]
  e_total_bed = torch.minimum(e_total[range_bed1[0]:range_bed1[1]], e_total[range_bed2[0]:range_bed2[1]])
  e_total_windoor = torch.maximum(e_total[range_windoor1[0]:range_windoor1[1]], e_total[range_windoor2[0]:range_windoor2[1]])

  sm = torch.nn.Softmax(dim=0)
  e_total = torch.cat([e_total_default,e_total_bed,e_total_windoor])
  epsilon = torch.exp(torch.as_tensor([-5])).to(device)
  if use_log:
    e_total = -torch.log(1.0 + epsilon - e_total)
  e_total = e_total*sm(e_total*10.0)
  e_total = torch.sum(e_total)
  return e_total.view(1,)

def eval_access(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun,use_log=True,device='cpu'):
  """
  Evaluates the Accessibility loss for each predicted token value
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
  is_sit = torch.sum(torch.eq(dict_cat2fun['sit'],categories),-1) > 0
  is_bed = torch.sum(torch.eq(dict_cat2fun['bed'],categories),-1) > 0
  is_table = torch.sum(torch.eq(dict_cat2fun['table'],categories),-1) > 0
  is_near_ceil = torch.sum(torch.eq(dict_cat2fun['near_ceil'],categories),-1) > 0
  is_tall = torch.sum(torch.eq(dict_cat2fun['tall'],categories),-1) > 0
  is_door = torch.sum(torch.eq(dict_cat2fun['door'],categories),-1) > 0
  is_window = torch.sum(torch.eq(dict_cat2fun['window'],categories),-1) > 0
  is_windoor = torch.sum(torch.eq(dict_cat2fun['windoor'],categories),-1) > 0
  mat_supp = (is_supporter.view(-1,1) * is_supported.view(1,-1)) + (is_supported.view(-1,1) * is_supporter.view(1,-1)) == 0
  mat_sittable = (is_sit.view(-1,1) * is_table.view(1,-1)) + (is_table.view(-1,1) * is_sit.view(1,-1)) == 0
  mat_wintall = (is_window.view(-1,1) * is_tall.view(1,-1)) + (~is_window.view(-1,1) * ~is_window.view(1,-1))
  mat_ceil = ~is_near_ceil.view(-1,1) * ~is_near_ceil.view(1,-1)
  coll_mat = mat_supp * mat_sittable * mat_ceil * mat_wintall
  inv_eye = ~torch.eye(coll_mat.size(0),coll_mat.size(0),dtype=torch.bool)
  coll_mat = ~coll_mat[inv_eye].view(n_rows,n_rows-1)
  coll_room = ((is_windoor + is_near_ceil) > 0).view(-1,1)
  coll_mat = torch.cat([coll_room, coll_mat], dim=1).repeat(5,1)

  ind_mat = torch.zeros_like(value_mat,dtype=torch.bool)
  ind_mat[elem_ind,row_ind,:] = 1
  matA = value_mat[ind_mat].view(n_elem,1,5)
  matB = value_mat[~ind_mat].view(n_elem,-1,5)

  is_default = ~torch.logical_or(torch.logical_or(is_door,is_window),is_bed)
  is_default = is_default.repeat(5)
  all_pos = matA[is_default,:,0:2]
  all_dim = matA[is_default,:,2:4]
  all_rot = matA[is_default,:,4].view(-1,1,1)
  all_front_dir = torch.cat([-torch.sin(all_rot),torch.cos(all_rot)],dim=-1)
  all_front = torch.cat([-torch.sin(all_rot),torch.cos(all_rot)],dim=-1) * 0.5 * all_dim[:,:,[1]]
  all_side = torch.cat([torch.cos(all_rot),torch.sin(all_rot)],dim=-1) * 0.5 * all_dim[:,:,[0]]
  is_bed = is_bed.repeat(5)
  bed_pos = matA[is_bed,:,0:2]
  bed_dim = matA[is_bed,:,2:4]
  bed_rot = matA[is_bed,:,4].view(-1,1,1)
  bed_front = torch.cat([-torch.sin(bed_rot),torch.cos(bed_rot)],dim=-1) * 0.5 * bed_dim[:,:,[1]]
  bed_side_dir = torch.cat([torch.cos(bed_rot),torch.sin(bed_rot)],dim=-1)
  bed_side = bed_side_dir * 0.5 * bed_dim[:,:,[0]]
  is_windoor = is_windoor.repeat(5)
  windoor_pos = matA[is_windoor,:,0:2]
  windoor_dim = matA[is_windoor,:,2:4]
  windoor_rot = matA[is_windoor,:,4].view(-1,1,1)
  windoor_front_dir = torch.cat([-torch.sin(windoor_rot),torch.cos(windoor_rot)],dim=-1)
  windoor_front = windoor_front_dir * 0.5 * windoor_dim[:,:,[1]]
  windoor_side = torch.cat([torch.cos(windoor_rot),torch.sin(windoor_rot)],dim=-1) * 0.5 * windoor_dim[:,:,[0]]

  default_cornersA = torch.stack([all_pos + all_front + all_side, all_pos + all_front + all_side + 0.5 * all_front_dir,
                                      all_pos + all_front - all_side, all_pos + all_front - all_side + 0.5 * all_front_dir],dim=0)
  bed_corners1 = torch.stack([bed_pos + 0.5 * bed_front + bed_side, bed_pos + 0.5 * bed_front + bed_side + 0.5 * bed_side_dir,
                              bed_pos - 0.5 * bed_front + bed_side, bed_pos - 0.5 * bed_front + bed_side + 0.5 * bed_side_dir],dim=0)
  bed_corners2 = torch.stack([bed_pos + 0.5 * bed_front - bed_side, bed_pos + 0.5 * bed_front - bed_side - 0.5 * bed_side_dir,
                              bed_pos - 0.5 * bed_front - bed_side, bed_pos - 0.5 * bed_front - bed_side - 0.5 * bed_side_dir],dim=0)
  windoor_corners1 = torch.stack([windoor_pos + windoor_front + windoor_side, windoor_pos + windoor_front + windoor_side + 0.5 * windoor_front_dir,
                                  windoor_pos + windoor_front - windoor_side, windoor_pos + windoor_front - windoor_side + 0.5 * windoor_front_dir],dim=0)
  windoor_corners2 = torch.stack([windoor_pos - windoor_front + windoor_side, windoor_pos - windoor_front + windoor_side - 0.5 * windoor_front_dir,
                                  windoor_pos - windoor_front - windoor_side, windoor_pos - windoor_front - windoor_side - 0.5 * windoor_front_dir],dim=0)

  bboxes_default = torch.stack([torch.min(default_cornersA,0)[0],torch.max(default_cornersA,0)[0]],dim=2)
  bbboxes_bed1 = torch.stack([torch.min(bed_corners1,0)[0],torch.max(bed_corners1,0)[0]],dim=2)
  bbboxes_bed2 = torch.stack([torch.min(bed_corners2,0)[0],torch.max(bed_corners2,0)[0]],dim=2)
  bbboxes_windoor1 = torch.stack([torch.min(windoor_corners1,0)[0],torch.max(windoor_corners1,0)[0]],dim=2)
  bbboxes_windoor2 = torch.stack([torch.min(windoor_corners2,0)[0],torch.max(windoor_corners2,0)[0]],dim=2)
  bboxesA = torch.cat([bboxes_default,bbboxes_bed1,bbboxes_bed2,bbboxes_windoor1,bbboxes_windoor2],dim=0)
  coll_mat_full = torch.cat([coll_mat[is_default,:],coll_mat[is_bed,:],coll_mat[is_bed,:],coll_mat[is_windoor,:],coll_mat[is_windoor,:]],dim=0)
  range_def = [0,bboxes_default.size(0)]
  range_bed1 = [range_def[1],range_def[1]+bbboxes_bed1.size(0)]
  range_bed2 = [range_bed1[1],range_bed1[1]+bbboxes_bed2.size(0)]
  range_windoor1 = [range_bed2[1],range_bed2[1]+bbboxes_windoor1.size(0)]
  range_windoor2 = [range_windoor1[1],range_windoor1[1]+bbboxes_windoor2.size(0)]

  all_posB = matB[:,:,0:2]
  all_dimB = matB[:,:,2:4]
  all_rotB = matB[:,:,4].view(n_elem,-1,1)
  all_frontB = torch.cat([-torch.sin(all_rotB),torch.cos(all_rotB)],dim=-1) * 0.5 * all_dimB[:,:,[1]]
  all_sideB = torch.cat([torch.cos(all_rotB),torch.sin(all_rotB)],dim=-1) * 0.5 * all_dimB[:,:,[0]]
  all_cornersB = torch.stack([all_posB + all_frontB + all_sideB, all_posB + all_frontB - all_sideB, all_posB - all_frontB - all_sideB, all_posB - all_frontB + all_sideB],dim=0)
  min_cornersB = torch.min(all_cornersB,0)[0]
  max_cornersB = torch.max(all_cornersB,0)[0]
  bboxesB = torch.stack([min_cornersB,max_cornersB],dim=2)
  bboxesB = torch.cat([bboxesB[is_default,:,:,:],bboxesB[is_bed,:,:,:],bboxesB[is_bed,:,:,:],bboxesB[is_windoor,:,:,:],bboxesB[is_windoor,:,:,:]],dim=0)

  bbox_room = torch.stack([pos_room-dim_room/2,pos_room+dim_room/2],dim=0).view(1,1,2,2)
  e_room = aabb_iou_par(bbox_room, bboxesA.permute(1,0,2,3),invert=True).view(-1,1)
  e_objects = aabb_iou_par(bboxesA, bboxesB)

  e_total = torch.cat([e_room,e_objects],dim=1)
  e_total[coll_mat_full] = torch.zeros(1,)
  e_total = torch.sum(e_total,dim=1)

  e_total_default = e_total[range_def[0]:range_def[1]]
  e_total_bed = torch.minimum(e_total[range_bed1[0]:range_bed1[1]], e_total[range_bed2[0]:range_bed2[1]])
  e_total_windoor = torch.maximum(e_total[range_windoor1[0]:range_windoor1[1]], e_total[range_windoor2[0]:range_windoor2[1]])

  sm = torch.nn.Softmax(dim=0)
  e_total = torch.cat([e_total_default.view(-1,5),e_total_bed.view(-1,5),e_total_windoor.view(-1,5)],dim=0)
  epsilon = torch.exp(torch.as_tensor([-5])).to(device)
  if use_log:
    e_total = -torch.log(1.0 + epsilon - e_total)
  e_total = e_total*sm(e_total*10.0)
  e_total = torch.sum(e_total)
  return e_total.view(1,)
  
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

  all_cornersA = torch.stack([all_pos + all_front + all_side, all_pos + all_front - all_side, all_pos - all_front - all_side, all_pos - all_front + all_side],dim=0)
  min_cornersA = torch.min(all_cornersA,0)[0]
  max_cornersA = torch.max(all_cornersA,0)[0]
  bboxesA = torch.stack([min_cornersA,max_cornersA],dim=2)

  all_posB = matB[:,:,0:2]
  all_dimB = matB[:,:,2:4]
  all_rotB = matB[:,:,4].view(n_elem,-1,1)
  all_frontB = torch.cat([-torch.sin(all_rotB),torch.cos(all_rotB)],dim=-1) * 0.5 * all_dimB[:,:,[1]]
  all_sideB = torch.cat([torch.cos(all_rotB),torch.sin(all_rotB)],dim=-1) * 0.5 * all_dimB[:,:,[0]]
  all_cornersB = torch.stack([all_posB + all_frontB + all_sideB, all_posB + all_frontB - all_sideB, all_posB - all_frontB - all_sideB, all_posB - all_frontB + all_sideB],dim=0)
  min_cornersB = torch.min(all_cornersB,0)[0]
  max_cornersB = torch.max(all_cornersB,0)[0]
  bboxesB = torch.stack([min_cornersB,max_cornersB],dim=2)

  bbox_room = torch.stack([pos_room-dim_room/2,pos_room+dim_room/2],dim=0).view(1,1,2,2)
  e_room = aabb_iou_par(bbox_room, bboxesA.permute(1,0,2,3),invert=True).flatten()
  e_room[is_windoor.repeat(5)] = torch.zeros(1,)

  e_objects = aabb_iou_par(bboxesA, bboxesB)
  e_objects[coll_check_mat.repeat(5,1)] = torch.zeros(1,)
  e_total = torch.sum(e_objects) + torch.sum(e_room)
  return e_total.view(1,)

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

  all_cornersA = torch.stack([all_pos + all_front + all_side, all_pos + all_front - all_side, all_pos - all_front - all_side, all_pos - all_front + all_side],dim=0)
  min_cornersA = torch.min(all_cornersA,0)[0]
  max_cornersA = torch.max(all_cornersA,0)[0]
  bboxesA = torch.stack([min_cornersA,max_cornersA],dim=2)

  all_posB = matB[:,:,1:3]
  all_dimB = matB[:,:,3:5]
  all_rotB = matB[:,:,5].view(n_elem,-1,1)
  all_frontB = torch.cat([-torch.sin(all_rotB),torch.cos(all_rotB)],dim=-1) * 0.5 * all_dimB[:,:,[1]]
  all_sideB = torch.cat([torch.cos(all_rotB),torch.sin(all_rotB)],dim=-1) * 0.5 * all_dimB[:,:,[0]]
  all_cornersB = torch.stack([all_posB + all_frontB + all_sideB, all_posB + all_frontB - all_sideB, all_posB - all_frontB - all_sideB, all_posB - all_frontB + all_sideB],dim=0)
  min_cornersB = torch.min(all_cornersB,0)[0]
  max_cornersB = torch.max(all_cornersB,0)[0]
  bboxesB = torch.stack([min_cornersB,max_cornersB],dim=2)

  bbox_room = torch.stack([pos_room-dim_room/2,pos_room+dim_room/2],dim=0).view(1,1,2,2)
  e_room = aabb_iou_par(bbox_room, bboxesA.permute(1,0,2,3),invert=True).flatten()
  e_room[is_windoor] = torch.zeros(1,)

  e_objects = aabb_iou_par(bboxesA, bboxesB)
  e_objects[coll_check_mat] = torch.zeros(1,)
  e_total = torch.sum(e_objects) + torch.sum(e_room)
  return e_total.view(1,)
  
def evaluate_scenes(sequences, minvalue_dict, maxvalue_dict, dict_cat2fun, res=256, use_log=True, device='cpu', use_alt_loss=False):
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
        unquantized = unquantize_sequence(room_seq,minvalue_dict,maxvalue_dict,res,device=device)
        if use_alt_loss:
          seq_ergo_scores.append(score_overlap(unquantized, minvalue_dict,maxvalue_dict,res,dict_cat2fun,use_log=use_log,device=device))
        else:
          seq_ergo_scores.append(score_read_book(unquantized, minvalue_dict,maxvalue_dict,res,dict_cat2fun,use_log=use_log,device=device))
          seq_ergo_scores.append(score_watch_tv(unquantized, minvalue_dict,maxvalue_dict,res, dict_cat2fun,use_log=use_log,device=device))
          seq_ergo_scores.append(score_use_computer(unquantized, minvalue_dict,maxvalue_dict,res, dict_cat2fun,use_log=use_log,device=device))
          seq_ergo_scores.append(score_work_table(unquantized, minvalue_dict,maxvalue_dict,res, dict_cat2fun,use_log=use_log,device=device))
          seq_ergo_scores.append(score_access(unquantized, minvalue_dict,maxvalue_dict,res, dict_cat2fun,use_log=use_log,device=device))
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
        ergo_scores.extend(evaluate_scenes(sequences, minvalue_dict, maxvalue_dict, dict_cat2fun, res=res, use_log=True, device='cpu',use_alt_loss=use_alt_loss))

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