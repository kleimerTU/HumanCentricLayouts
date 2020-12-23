clear;clc;
util;
addpath(genpath('jsonlab-master'));
dataset = {};

%% pre-defined paramters
walllabel = 'wall';
floorlabel = 'floor';

%% load the scene list
scenedir = dir(house_folder);
for i = 3:length(scenedir)
    scenelist{i-2} = scenedir(i).name;
end

%% binary label vectors
labelset = load(labelsetname); % 'entirelabelset.mat' is a list of all fine-grained labels
labelset = labelset.labelset;

%% process each scene(house)
for i = 1:length(scenelist)
    scene_name = scenelist{i};
    filename = [house_folder, filesep, scene_name];
    filename = [filename, filesep, 'house.json'];
    % load the scene as a list of rooms
    try
        roomlist = parsehouse(filename); 
        [ roomlist ] = query_obj_metadata( roomlist, scene_name ); % fill in the object info (label, orientation, etc)
    catch
       fprintf([scene_name,': ERROR in parsing house.json!\n']);
       continue;
    end

    for j = 1:length(roomlist)
        room = roomlist{j};
        room_name = room.roomname;
        roompath = [room_folder, filesep, room.scenename, filesep, room.modelId];
        [obb_list,label_list] = get_obb_list(roompath);
        outpath_house = [wcf_folder, filesep, room.scenename];
        outpath_room = [wcf_folder, filesep, room.scenename, filesep, room_name];
        if ~isfile(outpath_house)
            mkdir(outpath_house);
        end
        save(outpath_room,'obb_list','label_list','scene_name','room_name');
    end
end

%%
function [obb_list, label_list] = get_obb_list(roompath)
ceilpath = [roompath 'c.obj'];
floorpath = [roompath 'f.obj'];
wallpath = [roompath 'w.obj'];
if isfile(wallpath)
    wall_objs = readOBJs(wallpath);
else
    wall_objs = {};
end
obb_list = zeros(9,size(wall_objs,1)+2);
bbox_room = [Inf Inf Inf; -Inf -Inf -Inf];

if isfile(ceilpath)
    ceil_objs = readOBJs(ceilpath);
    V_ceil = ceil_objs{1}.V;
    bbox_ceil = [min(V_ceil,[],1); max(V_ceil,[],1)];
    center_ceil = sum(bbox_ceil,1) / 2;
    size_ceil = bbox_ceil(2,:) - bbox_ceil(1,:);
    obb_list(1:3,1) = center_ceil;
    obb_list(4:6,1) = [0;1;0];
    obb_list(7:9,1) = size_ceil;
end

if isfile(floorpath)
    floor_objs = readOBJs(floorpath);
    V_floor = floor_objs{1}.V;
    bbox_floor = [min(V_floor,[],1); max(V_floor,[],1)];
    center_floor = sum(bbox_floor,1) / 2;
    size_floor = bbox_floor(2,:) - bbox_floor(1,:);
    obb_list(1:3,2) = center_floor;
    obb_list(4:6,2) = [0;1;0];
    obb_list(7:9,2) = size_floor;
end
label_list = cell(1,size(wall_objs,1+2));
label_list{1} = 'Ceiling';
label_list{2} = 'Floor';

for i=1:size(wall_objs,1)
    V = wall_objs{i}.V;
    bbox_wall = [min(V,[],1); max(V,[],1)];
    center_wall = sum(bbox_wall,1) / 2;
    size_wall = bbox_wall(2,:) - bbox_wall(1,:);
    bbox_room = [min(bbox_room(1,:),bbox_wall(1,:)); max(bbox_room(2,:),bbox_wall(2,:))];
    obb_list(1:3,i+2) = center_wall;
    obb_list(7:9,i+2) = size_wall;
end
center_room = sum(bbox_room,1) / 2;
for i=1:size(wall_objs,1)
    centervec = center_room - obb_list(1:3,i+2)';
    [~,imax] = max(abs(centervec));
    obb_list(imax+3,i+2) = sign(centervec(imax));
    if imax == 3 % for some reason we need to swap x and z for these walls
        obb_list(7:9,i+2) = [obb_list(9,i+2); obb_list(8,i+2); obb_list(7,i+2)];
    end
    label_list{i+2} = ['Wall_' num2str(i)];
end
end