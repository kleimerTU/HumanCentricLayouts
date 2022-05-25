import pickle
import time
import torch
from torch.utils.data import Dataset, DataLoader
import transformers

from src import utils
from src.modeling_gpt2_custom import GPT2LMHeadModelCustom
from src.main_functions import *

class CustomDataset(Dataset):
  """
  A simple custom Dataset class for storing the layouts and corresponding ergonomic scores
  """
  def __init__(self, sequences, max_len, res, ergo_weights=[], room_masks=[]):
    self.sequences = sequences
    self.indices = []
    if len(ergo_weights) < len(sequences):
      self.ergo_weights = [0.0] * len(sequences)
    else:
      self.ergo_weights = ergo_weights
    for seq in self.sequences:
      self.indices.append([i % 6 for i in range(len(seq))])
      n_pad = max_len - len(seq)
      if n_pad > 0:
        seq.extend([res] * n_pad)
        self.indices[-1].extend([res] * n_pad)
      for i in range(len(seq)):
        if (seq[i] == res):
          self.indices[-1][i] = max_len-1
    self.nsequences = len(self.sequences)
    self.positions = [i for i in range(max_len)]
    self.room_masks = room_masks
    return

  def __len__(self):
    return self.nsequences

  def __getitem__(self, item):
    return_dict = {'text':self.sequences[item],
                   'label':self.sequences[item],
                   'position': self.positions,
                   'indices': self.indices[item],
                   'ergo_weights': self.ergo_weights[item]}
    if len(self.room_masks) > 0:
      return_dict['room_mask'] = self.room_masks[item]
    return return_dict

class CustomDatasetRand(Dataset):
  """
  A custom Dataset class that uses only one randomly chosen augmented variation of each layout per epoch instead of using them all
  """
  def __init__(self, sequences, max_len, res, ergo_weights=[], room_masks=[]):
    self.sequences = sequences
    self.indices = []
    if len(ergo_weights) < len(sequences):
      self.ergo_weights = [0.0] * len(sequences)
    else:
      self.ergo_weights = ergo_weights
    for seq in self.sequences:
      self.indices.append([i % 6 for i in range(len(seq))])
      n_pad = max_len - len(seq)
      if n_pad > 0:
        seq.extend([res] * n_pad)
        self.indices[-1].extend([res] * n_pad)
      for i in range(len(seq)):
        if (seq[i] == res):
          self.indices[-1][i] = max_len-1
    self.nsequences = int(len(self.sequences)/8)
    self.positions = [i for i in range(max_len)]
    self.room_masks = room_masks
    return

  def __len__(self):
    return self.nsequences

  def __getitem__(self, item):
    item = item * 8 + torch.randint(0,8,(1,))
    return_dict = {'text':self.sequences[item],
                   'label':self.sequences[item],
                   'position': self.positions,
                   'indices': self.indices[item],
                   'ergo_weights': self.ergo_weights[item]}
    if len(self.room_masks) > 0:
      return_dict['room_mask'] = self.room_masks[item]
    return return_dict
    
class CustomDatasetPermute(Dataset):
  """
  A custom Dataset class that uses one randomly chosen and permuted augmented variation of each layout per epoch
  """
  def __init__(self, sequences, max_len, res, ergo_weights=[], room_masks=[]):
    self.sequences = sequences
    self.indices = []
    self.max_len = max_len
    if len(ergo_weights) < len(sequences):
      self.ergo_weights = [0.0] * len(sequences)
    else:
      self.ergo_weights = ergo_weights
    for seq in self.sequences:
      self.indices.append([i % 6 for i in range(len(seq))])
      n_pad = max_len - len(seq)
      if n_pad > 0:
        seq.extend([res] * n_pad)
        self.indices[-1].extend([res] * n_pad)
      for i in range(len(seq)):
        if (seq[i] == res):
          self.indices[-1][i] = max_len-1
    self.nsequences = int(len(self.sequences)/8)
    self.positions = [i for i in range(max_len)]
    self.room_masks = room_masks
    return

  def __len__(self):
    return self.nsequences

  def __getitem__(self, item):
    item = item * 8 + torch.randint(0,8,(1,))
    sequence = torch.as_tensor(self.sequences[item])
    end_token = sequence[-1].item()
    sequence = sequence[:-1].view(-1,6)
    post_windoor = torch.logical_and(sequence[:,0] > 2,sequence[:,0] < self.max_len)
    random_order = torch.randperm(torch.sum(post_windoor))
    subseq = sequence[post_windoor,:]
    subseq = subseq[random_order,:]
    sequence[post_windoor,:] = subseq
    sequence = sequence.flatten().tolist()
    sequence.append(end_token)
    return_dict = {'text':sequence,
                   'label':sequence,
                   'position': self.positions,
                   'indices': self.indices[item],
                   'ergo_weights': self.ergo_weights[item]}
    if len(self.room_masks) > 0:
      return_dict['room_mask'] = self.room_masks[item]
    return return_dict

class CustomCollator(object):
  """
  A simple custom Collator class for picking items stored in the dataset
  """
  def __init__(self):
      return

  def __call__(self, sequences):
    texts = [sequence['text'] for sequence in sequences]
    labels = [sequence['label'] for sequence in sequences]
    positions = [sequence['position'] for sequence in sequences]
    indices = [sequence['indices'] for sequence in sequences]
    ergo_weights = [sequence['ergo_weights'] for sequence in sequences]
    room_masks = [sequence['room_mask'] for sequence in sequences if 'room_mask' in sequence.keys()]
    inputs = {'input_ids':torch.as_tensor(labels),'labels':torch.as_tensor(labels), 'position_ids': torch.as_tensor(positions), 'index_ids': torch.as_tensor(indices), 'ergo_weights': torch.as_tensor(ergo_weights)}
    if len(room_masks) > 0:
      inputs['room_masks'] = torch.stack(room_masks,dim=0).float()
    return inputs
    
class ResNet18(nn.Module):
  """
  Build a feature extractor using the ResNet18 architecture for image based inputs.
  """
  def __init__(self, freeze_bn=False, input_channels=1, feature_size=64):
    super(ResNet18, self).__init__()
    self._feature_size = feature_size

    self._feature_extractor = models.resnet18(pretrained=False)

    self._feature_extractor.conv1 = torch.nn.Conv2d(input_channels,64,kernel_size=(11, 11),stride=(4, 4),padding=(2, 2),bias=False)

    self._feature_extractor.fc = nn.Sequential(nn.Linear(512, 512), nn.ReLU(),nn.Linear(512, self.feature_size))
    self._feature_extractor.avgpool = nn.AdaptiveAvgPool2d((1, 1))

  @property
  def feature_size(self):
      return self._feature_size

  def forward(self, X):
      return self._feature_extractor(X)

class AlexNet(nn.Module):
  """
  Build a feature extractor using the AlexNet architecture for image based inputs.
  """
  def __init__(self, input_channels, feature_size):
    super(AlexNet, self).__init__()
    self._feature_size = feature_size

    self._feature_extractor = models.alexnet(pretrained=False)
    self._feature_extractor.features[0] = torch.nn.Conv2d(input_channels,64,kernel_size=(11, 11),stride=(4, 4),padding=(2, 2))

    self._fc = nn.Linear(9216, self._feature_size)

  def forward(self, X):
    X = self._feature_extractor.features(X)
    X = self._feature_extractor.avgpool(X)
    X = self._fc(X.view(X.shape[0], -1))

    return X

  @property
  def feature_size(self):
    return self._feature_size

class RoomEncoder(nn.Module):
  """Build a feature extractor using the pretrained ResNet18 architecture for
  image based inputs.
  """
  def __init__(self, freeze_bn=False, input_channels=1, feature_size=64, res=256):
    super().__init__()
    #self.extractor = ResNet18(freeze_bn=freeze_bn,input_channels=input_channels,feature_size=feature_size)
    self.extractor = AlexNet(input_channels=input_channels,feature_size=feature_size)
    self.fc_room = nn.Linear(self.extractor.feature_size, res)

  def forward(self, X):
      return self.fc_room(self.extractor(X))

def train(dataloader, model, optimizer_, scheduler_, device_, dict_int2cat, minvalue_dict, maxvalue_dict, res, max_furniture, dict_cat2fun, use_ergo=False, use_weight=False, eval_ergo_loss=False, order_switch=0, use_log=True, alt_loss=False, room_encoder=None, room_mode='replace_angle_after',pred_sigma=8.0, pred_random=False):
  """
  Function that trains the Transformer for one epoch
  """
  #global model
  seq_len = max_furniture * 6 + 6
  total_loss = 0.0
  total_ergo_loss = 0.0
  total_loss_weighted = 0.0
  total_ergo_loss_weighted = 0.0
  batch_loss_hist = []
  batch_ergo_loss_hist = []
  model.train()
  if room_encoder != None:
    room_encoder.train()
    posmap = torch.stack([torch.linspace(0.0,1.0,256).view(256,1).expand(-1,256),
                          torch.linspace(0.0,1.0,256).view(1,256).expand(256,-1)],dim=0)
  nvalid = 0
  loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
  cur_batch = 0
  dict_cat2fun_cuda = dict_cat2fun.copy()
  for key in dict_cat2fun_cuda.keys():
      dict_cat2fun_cuda[key] = dict_cat2fun_cuda[key].to(device_)
  for batch in dataloader:
    loss_sum = 0
    ergo_weights = batch.pop('ergo_weights',None).to(device_)
    room_masks = batch.pop('room_masks',None)
    batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
    labels = batch.pop('labels',None) # remove labels since we want to compute the loss manually
    model.zero_grad()
    if room_encoder and room_masks != None:
      room_encoder.zero_grad()
      room_masks = room_masks.to(device_)
      posmap = posmap.to(device_)
      if "all" in room_mode:
        batch['room_encodings'] = room_encoder(torch.cat([room_masks,posmap.view(2,256,256).repeat([room_masks.size(0),1,1,1])],dim=1)).view(-1,5,res)
      else:
        batch['room_encodings'] = room_encoder(torch.cat([room_masks,posmap.view(2,256,256).repeat([room_masks.size(0),1,1,1])],dim=1))
      batch['room_mode'] = room_mode
    outputs = model(**batch)
    logits = outputs[0]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_per_logit = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    batch_ergo_loss = 0
    batch_ergo_loss_weighted = 0
    n_ergo_valid = 0

    loss_per_logit = loss_per_logit.view(-1,seq_len)
    loss_per_logit = torch.cat([torch.zeros(loss_per_logit.size(0),1).to(device_),loss_per_logit[:,:-1]],1)
    loss_per_logit = loss_per_logit.view(-1,max_furniture+1,6)
    
    if use_ergo or eval_ergo_loss:
      for i in range(labels.size(0)):
        ergo_loss = torch.zeros(1,device=device_)
        valid_losses = 0
        interp_val, label_mat = get_predictions(labels[i,:], logits[i,:,:],minvalue_dict,maxvalue_dict,res,sigma=pred_sigma,sample_random=pred_random,device=device_)
        interp_val[:,0] = label_mat[:,0]
        interp_val[0,:] = label_mat[0,:]
        if order_switch == 1:
          interp_val = interp_val[:,[0,4,5,2,3,1]]
          label_mat = label_mat[:,[0,4,5,2,3,1]]
        if order_switch == 2:
          interp_val = interp_val[:,[0,2,3,4,5,1]]
          label_mat = label_mat[:,[0,2,3,4,5,1]]
        is_valid = torch.sum(interp_val > (res-0.9),1) + torch.sum(label_mat > (res-0.9),1) < 1.0
        interp_val = interp_val[is_valid,:]
        label_mat = label_mat[is_valid,:]
        label_mat = unquantize_sequence(label_mat,minvalue_dict,maxvalue_dict,res,device=device_)
        interp_val = unquantize_sequence(interp_val,minvalue_dict,maxvalue_dict,res,device=device_)

        if alt_loss: # use overlap loss instead of ergo loss
          loss_book = torch.sum(eval_overlap(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun_cuda,device=device_))
          if loss_book > -1.0:
            ergo_loss = ergo_loss + loss_book
            valid_losses = valid_losses + 1
          else:
            ergo_loss = ergo_loss + 1.0 + use_log * 4.0
            valid_losses = valid_losses + 1
          if valid_losses > 0:
            ergo_loss = 1.0 * ergo_loss / valid_losses
          if not torch.isnan(ergo_loss):
            batch_ergo_loss = batch_ergo_loss + ergo_loss
            batch_ergo_loss_weighted = batch_ergo_loss_weighted + ergo_loss * ergo_weights[i]
            n_ergo_valid = n_ergo_valid + 1
        else:
          loss_book = torch.sum(eval_read_book(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun_cuda,use_log=use_log,device=device_))
          loss_tv = torch.sum(eval_watch_tv(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun_cuda,use_log=use_log,device=device_))
          loss_comp = torch.sum(eval_use_computer(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun_cuda,use_log=use_log,device=device_))
          loss_table = torch.sum(eval_work_table(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun_cuda,use_log=use_log,device=device_))
          loss_access = torch.sum(eval_access(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun_cuda,use_log=use_log,device=device_))
          if loss_book > -1.0:
            ergo_loss = ergo_loss + loss_book
            valid_losses = valid_losses + 1
          else:
            ergo_loss = ergo_loss + 1.0 + use_log * 4.0
            valid_losses = valid_losses + 1
          if loss_tv > -1.0:
            ergo_loss = ergo_loss + loss_tv
            valid_losses = valid_losses + 1
          if loss_comp > -1.0:
            ergo_loss = ergo_loss + loss_comp
            valid_losses = valid_losses + 1
          if loss_table > -1.0:
            ergo_loss = ergo_loss + loss_table
            valid_losses = valid_losses + 1
          if loss_access > -1.0:
            ergo_loss = ergo_loss + loss_access
            valid_losses = valid_losses + 1
          if valid_losses > 0:
            ergo_loss = 1.0 * ergo_loss / valid_losses
          if not torch.isnan(ergo_loss):
            batch_ergo_loss = batch_ergo_loss + ergo_loss
            batch_ergo_loss_weighted = batch_ergo_loss_weighted + ergo_loss * ergo_weights[i]
            n_ergo_valid = n_ergo_valid + 1
        
    loss = torch.mean(loss_per_logit)
    loss_weighted = torch.mean(loss_per_logit * (1.0-ergo_weights.view(-1,1,1)))
    if use_weight:
      loss_sum = loss_sum + loss_weighted
    else:
      loss_sum = loss_sum + loss
    if n_ergo_valid > 0:
      batch_ergo_loss = batch_ergo_loss / n_ergo_valid
      batch_ergo_loss_weighted = batch_ergo_loss_weighted / n_ergo_valid
      total_ergo_loss = total_ergo_loss + batch_ergo_loss.item()
      total_ergo_loss_weighted = total_ergo_loss_weighted + batch_ergo_loss_weighted.item()
      if use_ergo:
        if use_weight:
          loss_sum = loss_sum + batch_ergo_loss_weighted
        else:
          loss_sum = loss_sum + batch_ergo_loss
    if not torch.isnan(loss):
      total_loss = total_loss + loss.item()
      total_loss_weighted = total_loss_weighted + loss_weighted.item()
      nvalid += 1

    if nvalid > 0:
      batch_loss_hist.append(loss.item())
      if use_ergo:
        batch_ergo_loss_hist.append(ergo_loss.item())
    loss_sum.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if room_encoder:
      torch.nn.utils.clip_grad_norm_(room_encoder.parameters(), 1.0)
    optimizer_.step()
    scheduler_.step()
    cur_batch = cur_batch + 1
    print("Training batch", cur_batch, "of", len(dataloader), "- mean loss:", total_loss / nvalid, ", mean ergo loss:", total_ergo_loss / nvalid, end="\r")

  avg_epoch_loss = total_loss
  avg_ergo_loss = total_ergo_loss
  avg_epoch_loss_weighted = total_loss_weighted
  avg_ergo_loss_weighted = total_ergo_loss_weighted
  if nvalid > 0:
    avg_epoch_loss = avg_epoch_loss / nvalid
    avg_ergo_loss = avg_ergo_loss / nvalid
    avg_epoch_loss_weighted = avg_epoch_loss_weighted / nvalid
    avg_ergo_loss_weighted = avg_ergo_loss_weighted / nvalid
  return avg_epoch_loss, avg_ergo_loss, avg_epoch_loss_weighted, avg_ergo_loss_weighted

def validate(dataloader, model, device_, dict_int2cat, minvalue_dict, maxvalue_dict, res, max_furniture, dict_cat2fun, use_ergo=False, eval_ergo_loss=False, order_switch=0, use_log=True, alt_loss=False, room_encoder=None, room_mode='replace_angle_after', pred_sigma=8.0, pred_random=False):
  """
  Function that performs validation for the given Transformer model
  """
  #global model
  seq_len = max_furniture * 6 + 6
  predictions_labels = []
  true_labels = []
  total_loss = 0
  total_loss_weighted = 0
  total_ergo_loss = 0
  total_ergo_loss_weighted = 0
  cur_batch = 0

  model.eval()
  if room_encoder != None:
    room_encoder.eval()
    posmap = torch.stack([torch.linspace(0.0,1.0,256).view(256,1).expand(-1,256),
                          torch.linspace(0.0,1.0,256).view(1,256).expand(256,-1)],dim=0)
  nvalid = 0
  loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
  dict_cat2fun_cuda = dict_cat2fun.copy()
  for key in dict_cat2fun_cuda.keys():
      dict_cat2fun_cuda[key] = dict_cat2fun_cuda[key].to(device_)
  for batch in dataloader:
    ergo_weights = batch.pop('ergo_weights',None).to(device_)
    room_masks = batch.pop('room_masks',None)
    true_labels += batch['labels'].numpy().flatten().tolist()
    batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
    labels = batch.pop('labels',None)
    if room_encoder and room_masks != None:
      room_masks = room_masks.to(device_)
      posmap = posmap.to(device_)
      if "all" in room_mode:
        batch['room_encodings'] = room_encoder(torch.cat([room_masks,posmap.view(2,256,256).repeat([room_masks.size(0),1,1,1])],dim=1)).view(-1,5,res)
      else:
        batch['room_encodings'] = room_encoder(torch.cat([room_masks,posmap.view(2,256,256).repeat([room_masks.size(0),1,1,1])],dim=1))
      batch['room_mode'] = room_mode
    with torch.no_grad():
      outputs = model(**batch)
      logits = outputs[0]
      shift_logits = logits[..., :-1, :].contiguous()
      shift_labels = labels[..., 1:].contiguous()
      loss_per_logit = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
      batch_ergo_loss = 0
      batch_ergo_loss_weighted = 0
      n_ergo_valid = 0

      if use_ergo or eval_ergo_loss:
        for i in range(labels.size(0)):
          ergo_loss = torch.zeros(1,device=device_)
          valid_losses = 0
          interp_val, label_mat = get_predictions(labels[i,:], logits[i,:,:],minvalue_dict,maxvalue_dict,res,sigma=pred_sigma,sample_random=pred_random,device=device_)
          interp_val[:,0] = label_mat[:,0]
          interp_val[0,:] = label_mat[0,:]
          if order_switch == 1:
            interp_val = interp_val[:,[0,4,5,2,3,1]]
            label_mat = label_mat[:,[0,4,5,2,3,1]]
          if order_switch == 2:
            interp_val = interp_val[:,[0,2,3,4,5,1]]
            label_mat = label_mat[:,[0,2,3,4,5,1]]
          is_valid = torch.sum(interp_val > (res-0.9),1) + torch.sum(label_mat > (res-0.9),1) < 1.0
          interp_val = interp_val[is_valid,:]
          label_mat = label_mat[is_valid,:]
          label_mat = unquantize_sequence(label_mat,minvalue_dict,maxvalue_dict,res,device=device_)
          interp_val = unquantize_sequence(interp_val,minvalue_dict,maxvalue_dict,res,device=device_)

          if alt_loss: # use overlap loss instead of ergo loss
            loss_book = torch.sum(eval_overlap(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun_cuda,device=device_))
            if loss_book > -1.0:
              ergo_loss = ergo_loss + loss_book
              valid_losses = valid_losses + 1
            else:
              ergo_loss = ergo_loss + 1.0 + use_log * 4.0
              valid_losses = valid_losses + 1
            if valid_losses > 0:
              ergo_loss = 1.0 * ergo_loss / valid_losses
            if not torch.isnan(ergo_loss):
              batch_ergo_loss = batch_ergo_loss + ergo_loss
              batch_ergo_loss_weighted = batch_ergo_loss_weighted + ergo_loss * ergo_weights[i]
              n_ergo_valid = n_ergo_valid + 1
          else:
            loss_book = torch.sum(eval_read_book(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun_cuda,use_log=use_log,device=device_))
            loss_tv = torch.sum(eval_watch_tv(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun_cuda,use_log=use_log,device=device_))
            loss_comp = torch.sum(eval_use_computer(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun_cuda,use_log=use_log,device=device_))
            loss_table = torch.sum(eval_work_table(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun_cuda,use_log=use_log,device=device_))
            loss_access = torch.sum(eval_access(interp_val, label_mat,minvalue_dict,maxvalue_dict,res,dict_cat2fun_cuda,use_log=use_log,device=device_))
            if loss_book > -1.0:
              ergo_loss = ergo_loss + loss_book
              valid_losses = valid_losses + 1
            else:
              ergo_loss = ergo_loss + 1.0 + use_log * 4.0
              valid_losses = valid_losses + 1
            if loss_tv > -1.0:
              ergo_loss = ergo_loss + loss_tv
              valid_losses = valid_losses + 1
            if loss_comp > -1.0:
              ergo_loss = ergo_loss + loss_comp
              valid_losses = valid_losses + 1
            if loss_table > -1.0:
              ergo_loss = ergo_loss + loss_table
              valid_losses = valid_losses + 1
            if loss_access > -1.0:
              ergo_loss = ergo_loss + loss_access
              valid_losses = valid_losses + 1
            if valid_losses > 0:
              ergo_loss = 1.0 * ergo_loss / valid_losses
            if not torch.isnan(ergo_loss):
              batch_ergo_loss = batch_ergo_loss + ergo_loss
              batch_ergo_loss_weighted = batch_ergo_loss_weighted + ergo_loss * ergo_weights[i]
              n_ergo_valid = n_ergo_valid + 1
        
      loss = torch.mean(loss_per_logit)
      loss_weighted = torch.mean(loss_per_logit * (1.0-ergo_weights.view(-1,1,1)))
      if n_ergo_valid > 0:
        batch_ergo_loss = batch_ergo_loss / n_ergo_valid
        batch_ergo_loss_weighted = batch_ergo_loss_weighted / n_ergo_valid
        total_ergo_loss = total_ergo_loss + batch_ergo_loss.item()
        total_ergo_loss_weighted = total_ergo_loss_weighted + batch_ergo_loss_weighted.item()
      if not torch.isnan(loss):
        total_loss = total_loss + loss.item()
        total_loss_weighted = total_loss_weighted + loss_weighted.item()
        nvalid += 1
    cur_batch = cur_batch + 1
    print("Validation batch", cur_batch, "of", len(dataloader), "- mean loss:", total_loss / nvalid, ", mean ergo loss:", total_ergo_loss / nvalid,end="\r")
    
  avg_epoch_loss = total_loss
  avg_ergo_loss = total_ergo_loss
  avg_epoch_loss_weighted = total_loss_weighted
  avg_ergo_loss_weighted = total_ergo_loss_weighted
  if nvalid > 0:
    avg_epoch_loss = avg_epoch_loss / nvalid
    avg_ergo_loss = avg_ergo_loss / nvalid
    avg_epoch_loss_weighted = avg_epoch_loss_weighted / nvalid
    avg_ergo_loss_weighted = avg_ergo_loss_weighted / nvalid
  return true_labels, predictions_labels, avg_epoch_loss, avg_ergo_loss, avg_epoch_loss_weighted, avg_ergo_loss_weighted

def training_loop(training_params):
  """
  Trains a transformer model using the given training parameters
  """
  epochs = training_params["epochs"]
  model_name = training_params["model_name"]
  use_ergo = training_params["use_ergo"] # False for v0, v1 - True for v2, v3
  use_weight = training_params["use_weight"] # False for v0, v2 - True for v1, v3
  do_validate = training_params["do_validate"]
  continue_train = training_params["continue_train"]
  continue_epoch = training_params["continue_epoch"] # use -1 to load the latest saved epoch
  lr = training_params["lr"]
  batch_size = training_params["batch_size"]
  n_layer = training_params["n_layer"]
  n_head = training_params["n_head"]
  n_embd = training_params["n_embd"]
  eval_ergo_loss = training_params["eval_ergo_loss"]
  if torch.cuda.is_available():
    device = torch.device(training_params["device"])
  else:
    device = torch.device("cpu")
  
  print("Training", model_name)

  max_furniture = training_params["max_furniture"]
  max_seq_len = (max_furniture + 1) * 6 + 1
  res = training_params["res"]
  order_switch = training_params["order_switch"] # 0, 1 or 2, recommended is 1
  set_name = training_params["set_name"]
  suffix_name = training_params["suffix_name"]
  
  path_input_data = training_params["path_input_data"]
  path_trained_models = training_params["path_trained_models"]
  dict_cat2int, dict_int2cat, dict_cat2fun = utils.get_category_dicts(res)
  
  use_log = True
  if "use_log" in training_params.keys() and training_params["use_log"] != None:
    use_log = training_params["use_log"]

  use_init_model = False
  if "init_model" in training_params.keys() and training_params["init_model"] != None:
    use_init_model = True
    init_model = "model_" + training_params["init_model"]
    init_model_epoch = "best"
    if "init_model_epoch" in training_params.keys():
      init_model_epoch = training_params["init_model_epoch"]
    
  use_alt_loss = False
  if "use_alt_loss" in training_params.keys() and training_params["use_alt_loss"]:
    use_alt_loss = True
      
  pred_sigma = 1.0
  pred_random = False
  if "pred_sigma" in training_params.keys():
    pred_sigma = training_params["pred_sigma"]
  if "pred_random" in training_params.keys():
    pred_random = training_params["pred_random"]

  train_sequences = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_train_quantized" + suffix_name + ".pkl", "rb" ))
  val_sequences = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_val_quantized" + suffix_name + ".pkl", "rb" ))
  minvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_minvalues" + suffix_name + ".pkl", "rb" ))
  maxvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_maxvalues" + suffix_name + ".pkl", "rb" ))
  ergo_scores_train = pickle.load(open(path_input_data + set_name + "_data/" + set_name + "_ergo_scores_train" + suffix_name + ".pkl", "rb"))
  ergo_scores_val = pickle.load(open(path_input_data + set_name + "_data/" + set_name + "_ergo_scores_val" + suffix_name + ".pkl", "rb"))
  
  room_encoder = None
  room_masks_train = []
  room_masks_val = []
  room_mode = "replace_angle_after"
  room_fs = 64
  if "use_room_masks" in training_params.keys() and training_params["use_room_masks"]:
    if "room_mode" in training_params.keys() and training_params["room_mode"]:
      room_mode = training_params["room_mode"]
    room_enc_dim = 1
    if "all" in room_mode:
      room_enc_dim = 5
    if "room_fs" in training_params.keys() and training_params["room_fs"]:
      room_fs = training_params["room_fs"]
    room_encoder = RoomEncoder(freeze_bn=False, input_channels=3, feature_size=room_fs, res=room_enc_dim*256)
    room_masks_train = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_train_masks" + suffix_name + ".pkl", "rb" ))
    room_masks_val = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_val_masks" + suffix_name + ".pkl", "rb" ))

  if order_switch > 0:
    for i in range(len(train_sequences)):
      seq = torch.as_tensor(train_sequences[i]).view(-1,6)
      if order_switch == 1:
        seq = seq[:,[0,5,3,4,1,2]]
      if order_switch == 2:
        seq = seq[:,[0,5,1,2,3,4]]
      train_sequences[i] = seq.flatten().tolist()
    for i in range(len(val_sequences)):
      seq = torch.as_tensor(val_sequences[i]).view(-1,6)
      if order_switch == 1:
        seq = seq[:,[0,5,3,4,1,2]]
      if order_switch == 2:
        seq = seq[:,[0,5,1,2,3,4]]
      val_sequences[i] = seq.flatten().tolist()
      
  warmup_multiplier = 1
  if "augm_strategy" in training_params.keys():
    if training_params["augm_strategy"] == "subset":
      dataset = CustomDatasetRand(train_sequences[:], max_seq_len,res,ergo_weights=ergo_scores_train[:],room_masks=room_masks_train)
      dataset_val = CustomDatasetRand(val_sequences[:], max_seq_len,res,ergo_weights=ergo_scores_val[:],room_masks=room_masks_val)
      warmup_multiplier = 10
    elif training_params["augm_strategy"] == "permute":
      dataset = CustomDatasetPermute(train_sequences[:], max_seq_len,res,ergo_weights=ergo_scores_train[:],room_masks=room_masks_train)
      dataset_val = CustomDatasetPermute(val_sequences[:], max_seq_len,res,ergo_weights=ergo_scores_val[:],room_masks=room_masks_val)
      warmup_multiplier = 10
    else:
      dataset = CustomDataset(train_sequences[:], max_seq_len,res,ergo_weights=ergo_scores_train[:],room_masks=room_masks_train)
      dataset_val = CustomDataset(val_sequences[:], max_seq_len,res,ergo_weights=ergo_scores_val[:],room_masks=room_masks_val)
  else:
    dataset = CustomDataset(train_sequences[:], max_seq_len,res,ergo_weights=ergo_scores_train[:],room_masks=room_masks_train)
    dataset_val = CustomDataset(val_sequences[:], max_seq_len,res,ergo_weights=ergo_scores_val[:],room_masks=room_masks_val)

  collator = CustomCollator()
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
  dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collator)
  model_path = path_trained_models + "model_"  + model_name + "/"

  if continue_train:
    if continue_epoch < 0:
      continue_epoch = pickle.load(open(model_path + "cur_epoch.pkl", "rb")) - 1
    start_epoch = continue_epoch + 1
    checkpoint_path = model_path + "model_tmp" + str(continue_epoch) + "/"
    model = GPT2LMHeadModelCustom.from_pretrained(checkpoint_path)
    lr = pickle.load(open(checkpoint_path + "cur_lr.pkl", "rb"))[0]
    train_loss_hist = pickle.load(open(checkpoint_path + "train_loss_hist.pkl", "rb"))
    train_loss_weighted_hist = pickle.load(open(checkpoint_path + "train_loss_weighted_hist.pkl", "rb"))
    ergo_loss_hist = pickle.load(open(checkpoint_path + "ergo_loss_hist.pkl", "rb"))
    ergo_loss_weighted_hist = pickle.load(open(checkpoint_path + "ergo_loss_weighted_hist.pkl", "rb"))
    val_loss_hist = pickle.load(open(checkpoint_path + "val_loss_hist.pkl", "rb"))
    val_loss_weighted_hist = pickle.load(open(checkpoint_path + "val_loss_weighted_hist.pkl", "rb"))
    val_ergo_loss_hist = pickle.load(open(checkpoint_path + "val_ergo_loss_hist.pkl", "rb"))
    val_ergo_loss_weighted_hist = pickle.load(open(checkpoint_path + "val_ergo_loss_weighted_hist.pkl", "rb"))
    gen_loss_hist = pickle.load(open(checkpoint_path + "gen_loss_hist.pkl", "rb"))
  else:
    start_epoch = 0
    if use_init_model:
      if init_model_epoch == "best":
        val_loss_hist = pickle.load(open(path_trained_models + init_model + "/model_tmp9/val_loss_hist.pkl", "rb" ))
        n_epochs_skip = int(len(val_loss_hist) / 10)
        val_loss_hist_saved = val_loss_hist[n_epochs_skip-1::n_epochs_skip]
        epoch = torch.min(torch.as_tensor(val_loss_hist_saved),dim=0)[1].item()
      else:
        epoch = init_model_epoch
      model = GPT2LMHeadModelCustom.from_pretrained(path_trained_models + init_model + "/" + "model_tmp" + str(epoch) + "/")
    else:
      configuration = transformers.GPT2Config(vocab_size=res+1,n_embd=n_embd,n_ctx=max_seq_len,n_positions=max_seq_len, n_head=n_head, n_layer=n_layer)
      model = GPT2LMHeadModelCustom(configuration)
    train_loss_hist = []
    train_loss_weighted_hist = []
    ergo_loss_hist = []
    ergo_loss_weighted_hist = []
    val_loss_hist = []
    val_loss_weighted_hist = []
    val_ergo_loss_hist = []
    val_ergo_loss_weighted_hist = []

  total_steps = len(dataloader) * (epochs - start_epoch)
  if room_encoder:
    all_params = list(model.parameters())+list(room_encoder.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, eps=1e-8)
  else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8)
  scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=len(dataloader)*warmup_multiplier, num_training_steps=total_steps)

  for j in range(epochs - start_epoch):
    i = j + start_epoch
    print("Epoch", i+1, "of", epochs)
    model.to(device)
    if room_encoder:
      room_encoder.to(device)
    t_start = time.process_time()
    losses = train(dataloader, model, optimizer, scheduler, device, dict_int2cat, minvalue_dict, maxvalue_dict, res, max_furniture, dict_cat2fun, use_ergo=use_ergo, use_weight=use_weight, eval_ergo_loss=eval_ergo_loss, order_switch=order_switch, alt_loss=use_alt_loss, room_encoder=room_encoder, room_mode=room_mode, pred_sigma=pred_sigma,pred_random=pred_random,use_log=use_log)
    train_loss, ergo_loss, train_loss_weighted, ergo_loss_weighted, batch_loss_hist, batch_ergo_loss_hist = losses
    print("Training epoch", i+1, "of", epochs, "- mean loss:", train_loss, ", mean ergo loss:", ergo_loss, "Time:", time.process_time() - t_start, "s")
    train_loss_hist.append(train_loss)
    ergo_loss_hist.append(ergo_loss)
    train_loss_weighted_hist.append(train_loss_weighted)
    ergo_loss_weighted_hist.append(ergo_loss_weighted)
    if (i + 1) % int(epochs / 10) == 0:
      checkpoint_path = model_path + "model_tmp" + str(int((i + 1) / (epochs / 10) - 1)) + "/"
      model.save_pretrained(checkpoint_path)
      if room_encoder:
        torch.save(room_encoder.state_dict(),checkpoint_path+"room_encoder.pt")
      pickle.dump(i+1, open(model_path + "cur_epoch.pkl", "wb"))
      pickle.dump(scheduler.get_last_lr(), open(checkpoint_path + "cur_lr.pkl", "wb"))
      pickle.dump(train_loss_hist, open(checkpoint_path + "train_loss_hist.pkl", "wb"))
      pickle.dump(train_loss_weighted_hist, open(checkpoint_path + "train_loss_weighted_hist.pkl", "wb"))
      pickle.dump(ergo_loss_hist, open(checkpoint_path + "ergo_loss_hist.pkl", "wb"))
      pickle.dump(ergo_loss_weighted_hist, open(checkpoint_path + "ergo_loss_weighted_hist.pkl", "wb"))
      pickle.dump(batch_loss_hist, open(checkpoint_path + "batch_loss_hist.pkl", "wb"))
      pickle.dump(batch_ergo_loss_hist, open(checkpoint_path + "batch_ergo_loss_hist.pkl", "wb"))
    if do_validate:
      t_start = time.process_time()
      val_losses = validate(dataloader_val, model, device, dict_int2cat, minvalue_dict, maxvalue_dict, res, max_furniture, dict_cat2fun, use_ergo=use_ergo, eval_ergo_loss=eval_ergo_loss, order_switch=order_switch, alt_loss=use_alt_loss, room_encoder=room_encoder, room_mode=room_mode,use_log=use_log)[2:6]
      val_loss, val_ergo_loss, val_loss_weighted, val_ergo_loss_weighted = val_losses
      val_loss_hist.append(val_loss)
      val_ergo_loss_hist.append(val_ergo_loss)
      val_loss_weighted_hist.append(val_loss_weighted)
      val_ergo_loss_weighted_hist.append(val_ergo_loss_weighted)
      print("Validation epoch", i+1, "of", epochs, "- mean loss:", val_loss, ", mean ergo loss:", val_ergo_loss, "Time:", time.process_time() - t_start, "s")
    if (i + 1) % int(epochs / 10) == 0:
      pickle.dump(val_loss_hist, open(checkpoint_path + "val_loss_hist.pkl", "wb"))
      pickle.dump(val_loss_weighted_hist, open(checkpoint_path + "val_loss_weighted_hist.pkl", "wb"))
      pickle.dump(val_ergo_loss_hist, open(checkpoint_path + "val_ergo_loss_hist.pkl", "wb"))
      pickle.dump(val_ergo_loss_weighted_hist, open(checkpoint_path + "val_ergo_loss_weighted_hist.pkl", "wb"))
    
  plt.plot(train_loss_hist)
  plt.plot(val_loss_hist)
  plt.plot(ergo_loss_hist)

def ergo_validation_loop(training_params):
  """
  Evaluates the ergonomic loss for both training and validation data for each trained epoch of the model
  """
  epochs = training_params["epochs"]
  model_name = training_params["model_name"]
  use_ergo = training_params["use_ergo"] # False for v0, v1 - True for v2, v3
  use_weight = training_params["use_weight"] # False for v0, v2 - True for v1, v3
  batch_size = training_params["batch_size"]

  max_furniture = training_params["max_furniture"]
  max_seq_len = (max_furniture + 1) * 6 + 1
  res = training_params["res"]
  order_switch = training_params["order_switch"] # 0, 1 or 2, recommended is 1
  set_name = training_params["set_name"]
  suffix_name = training_params["suffix_name"]
  use_alt_loss = training_params["use_alt_loss"]
  if torch.cuda.is_available():
    device = torch.device(training_params["device"])
  else:
    device = torch.device("cpu")
  
  path_input_data = training_params["path_input_data"]
  path_trained_models = training_params["path_trained_models"]

  train_sequences = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_train_quantized" + suffix_name + ".pkl", "rb" ))
  val_sequences = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_val_quantized" + suffix_name + ".pkl", "rb" ))
  minvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_minvalues" + suffix_name + ".pkl", "rb" ))
  maxvalue_dict = pickle.load(open( path_input_data + set_name + "_data/" + set_name + "_maxvalues" + suffix_name + ".pkl", "rb" ))
  ergo_scores_train = pickle.load(open(path_input_data + set_name + "_data/" + set_name + "_ergo_scores_train" + suffix_name + ".pkl", "rb"))
  ergo_scores_val = pickle.load(open(path_input_data + set_name + "_data/" + set_name + "_ergo_scores_val" + suffix_name + ".pkl", "rb"))

  if order_switch > 0:
    for i in range(len(train_sequences)):
      seq = torch.as_tensor(train_sequences[i]).view(-1,6)
      if order_switch == 1:
        seq = seq[:,[0,5,3,4,1,2]]
      if order_switch == 2:
        seq = seq[:,[0,5,1,2,3,4]]
      train_sequences[i] = seq.flatten().tolist()
    for i in range(len(val_sequences)):
      seq = torch.as_tensor(val_sequences[i]).view(-1,6)
      if order_switch == 1:
        seq = seq[:,[0,5,3,4,1,2]]
      if order_switch == 2:
        seq = seq[:,[0,5,1,2,3,4]]
      val_sequences[i] = seq.flatten().tolist()

  collator = CustomCollator()
  dataset = CustomDataset(train_sequences[:], max_seq_len,res,ergo_weights=ergo_scores_train[:])
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
  dataset_val = CustomDataset(val_sequences[:], max_seq_len,res,ergo_weights=ergo_scores_val[:])
  dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collator)
  model_path = path_trained_models + "model_"  + model_name + "/"

  train_loss_hist = []
  train_loss_weighted_hist = []
  train_ergo_loss_hist = []
  train_ergo_loss_weighted_hist = []
  val_loss_hist = []
  val_loss_weighted_hist = []
  val_ergo_loss_hist = []
  val_ergo_loss_weighted_hist = []

  print("Validating", model_name)
  start_epoch = 0
  for j in range(epochs - start_epoch):
    i = j + start_epoch
    print("Epoch", i+1, "of", epochs)
    checkpoint_path = model_path + "model_tmp" + str(i) + "/"
    model = GPT2LMHeadModelCustom.from_pretrained(checkpoint_path)
    model.to(device)

    train_losses = validate(dataloader, model, device, dict_int2cat, minvalue_dict, maxvalue_dict, res, max_furniture, dict_cat2fun, use_ergo=use_ergo, order_switch=order_switch, alt_loss=use_alt_loss)[2:6]
    train_loss, train_ergo_loss, train_loss_weighted, train_ergo_loss_weighted = train_losses
    train_loss_hist.append(train_loss)
    train_loss_weighted_hist.append(train_loss_weighted)
    train_ergo_loss_hist.append(train_ergo_loss)
    train_ergo_loss_weighted_hist.append(train_ergo_loss_weighted)
    print("Training epoch", i+1, "of", epochs, "- mean loss:", train_loss, ", mean ergo loss:", ergo_loss, "Time:", time.process_time() - t_start, "s")
    pickle.dump(train_ergo_loss_hist, open(checkpoint_path + "ergo_loss_hist.pkl", "wb"))
    pickle.dump(train_ergo_loss_weighted_hist, open(checkpoint_path + "ergo_loss_weighted_hist.pkl", "wb"))

    val_losses = validate(dataloader_val, model, device, dict_int2cat, minvalue_dict, maxvalue_dict, res, max_furniture, dict_cat2fun, use_ergo=use_ergo, order_switch=order_switch, alt_loss=use_alt_loss)[2:6]
    val_loss, val_ergo_loss, val_loss_weighted, val_ergo_loss_weighted = val_losses
    val_loss_hist.append(val_loss)
    val_ergo_loss_hist.append(val_ergo_loss)
    val_loss_weighted_hist.append(val_loss_weighted)
    val_ergo_loss_weighted_hist.append(val_ergo_loss_weighted)
    print("Validation epoch", i+1, "of", epochs, "- mean loss:", val_loss, ", mean ergo loss:", val_ergo_loss, "Time:", time.process_time() - t_start, "s")
    pickle.dump(val_ergo_loss_hist, open(checkpoint_path + "val_ergo_loss_hist.pkl", "wb"))
    pickle.dump(val_ergo_loss_weighted_hist, open(checkpoint_path + "val_ergo_loss_weighted_hist.pkl", "wb"))
    
  plt.plot(train_loss_hist)
  plt.plot(val_loss_hist)
  plt.plot(train_ergo_loss_hist)
  plt.plot(val_ergo_loss_hist)