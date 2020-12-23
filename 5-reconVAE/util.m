scene_name = 'pretrained_scenes';
data_filename = [scene_name, '.mat'];

%% params to be set
data_folder = ['..', filesep, '0-data'];
input_folder = [data_folder, filesep, '4-pydata', filesep, 'output'];
labelset_folder = [data_folder, filesep, 'labelsets'];
output_folder = [data_folder, filesep, '5-generated_scenes', filesep, scene_name];
output_obj_folder = [output_folder, filesep, 'recon'];
output_img_folder = [output_folder, filesep, 'images'];
objfolder_path = [data_folder, filesep, 'SUNCG', filesep, 'object'];

labelset_filename = [labelset_folder,filesep,'bedroomlabelset.mat'];
objperlabel_filename = [data_folder,filesep,'object_per_label',filesep,'bedroom_objset_per_label.mat'];

addpath(genpath('..\3-datapreparation'));

if ~exist(output_folder,'dir')
   mkdir(output_folder);
end
if ~exist(output_obj_folder, 'dir')
	mkdir(output_obj_folder);
end
if ~exist(output_img_folder, 'dir')
	mkdir(output_img_folder);
end