import torch

category_valid = ['room','door','window','bed','single_bed','kids_bed','shaped_sofa','sofa','table','desk','coffee_table','side_table',
                  'wardrobe_cabinet','children_cabinet','dresser','dressing_table','sideboard','shelving','chair','lounge_chair','armchair','special_chair',
                  'barstool','ottoman','stand','tv_stand','television','computer','floor_lamp','indoor_lamp','chandelier']

def get_category_dicts(res):
    dict_cat2int = {}
    dict_int2cat = {}
    dict_cat2fun = {}
    
    dict_int2cat[str(res)] = "#" # stop token
    n_cat = 0
    for cat in category_valid:
      dict_cat2int[cat] = n_cat
      dict_int2cat[str(n_cat)] = cat
      n_cat += 1
      
    dict_cat2fun['light'] = torch.as_tensor([dict_cat2int[i] for i in ['window','floor_lamp','indoor_lamp','chandelier']]).view(1,-1)
    dict_cat2fun['glare'] = torch.as_tensor([dict_cat2int[i] for i in ['window','floor_lamp','indoor_lamp']]).view(1,-1)
    dict_cat2fun['sit'] = torch.as_tensor([dict_cat2int[i] for i in ['bed','single_bed','kids_bed','shaped_sofa','sofa','chair','lounge_chair',
                                                                    'armchair','special_chair','barstool','ottoman']]).view(1,-1)
    dict_cat2fun['bed'] = torch.as_tensor([dict_cat2int[i] for i in ['bed','single_bed','kids_bed']]).view(1,-1)
    dict_cat2fun['table'] = torch.as_tensor([dict_cat2int[i] for i in ['table','desk','dressing_table','coffee_table']]).view(1,-1)
    dict_cat2fun['computer'] = torch.as_tensor([dict_cat2int[i] for i in ['computer']]).view(1,-1)
    dict_cat2fun['tv'] = torch.as_tensor([dict_cat2int[i] for i in ['television']]).view(1,-1)
    dict_cat2fun['supporter'] = torch.as_tensor([dict_cat2int[i] for i in ['table','desk','coffee_table','side_table','dresser','dressing_table','sideboard','shelving','stand','tv_stand']]).view(1,-1)
    dict_cat2fun['supported'] = torch.as_tensor([dict_cat2int[i] for i in ['television','computer','indoor_lamp']]).view(1,-1)
    dict_cat2fun['near_ceil'] = torch.as_tensor([dict_cat2int[i] for i in ['chandelier']]).view(1,-1)
    dict_cat2fun['chair'] = torch.as_tensor([dict_cat2int[i] for i in ['chair','lounge_chair','special_chair','armchair','ottoman','shaped_sofa']]).view(1,-1)
    dict_cat2fun['sofa'] = torch.as_tensor([dict_cat2int[i] for i in ['shaped_sofa']]).view(1,-1)
    dict_cat2fun['tall'] = torch.as_tensor([dict_cat2int[i] for i in ['wardrobe_cabinet','children_cabinet','shelving']]).view(1,-1)
    dict_cat2fun['window'] = torch.as_tensor([dict_cat2int[i] for i in ['window']]).view(1,-1)
    dict_cat2fun['door'] = torch.as_tensor([dict_cat2int[i] for i in ['door']]).view(1,-1)
    dict_cat2fun['windoor'] = torch.as_tensor([dict_cat2int[i] for i in ['window','door']]).view(1,-1)
    return dict_cat2int, dict_int2cat, dict_cat2fun

def get_sampling_type(top_k,top_p,max_noncat,max_windoor):
    if top_k > 0:
      sampling_type = "k" + str(top_k)
    else:
      sampling_type = "p" + str(top_p).replace(".", "")
      if max_noncat:
        sampling_type = sampling_type + "max"
        if max_windoor:
          sampling_type = sampling_type + "wd"
    return sampling_type
    
def get_valid_categories():
    return category_valid