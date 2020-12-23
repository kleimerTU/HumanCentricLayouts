function [ output_args ] = plotRoom( p_data, scheme )

% scheme == 0: show objects with different color
% scheme == 1: show areas with different color

corlist = {[1,0,0,0.3],[0,1,0,0.3],[0,1,1,0.3],[0,0,1,0.3],[1,0,1,0.3]};

floor_ind = strmatch('floor',p_data.labellist,'exact');
ceiling_ind = strmatch('ceiling',p_data.labellist,'exact');
wall_ind = strmatch('wall',p_data.labellist,'exact');
com_ind = union(floor_ind,ceiling_ind);
com_ind = union(com_ind,wall_ind);
% idx = 1:length(p_data.labellist);
% idx = setdiff(idx,com_ind);
labellist = p_data.labellist;
obblist = p_data.obblist;

for i = 1:length(labellist)
    p = obblist(:,i);
    
    center = p(1:3);
    dir_1 = p(4:6);
    dir_2 = p(7:9);
    lengths = p(10:12);

    dir_1 = dir_1/norm(dir_1);
    dir_2 = dir_2/norm(dir_2);
    dir_3 = cross(dir_1,dir_2);
    dir_3 = dir_3/norm(dir_3); 

    d1 = 0.5*lengths(1)*dir_1;
    d2 = 0.5*lengths(2)*dir_2;
    d3 = 0.5*lengths(3)*dir_3;
    cornerpoints(1,:) = center-d1-d2-d3;
    cornerpoints(2,:) = center+d1-d2-d3;
    cornerpoints(3,:) = center-d1-d2+d3;
    cornerpoints(4,:) = center+d1-d2+d3;
    cornerpoints(:,3) = -cornerpoints(:,3);
    
    idx = find(com_ind==i);
    if(length(idx)==0)
        if(scheme==1&&isfield(p_data,'area_id'))
            aid = p_data.area_id(i);
            if(aid==0)
                cor = [0,0,0,0.3];
            else
                cor = corlist{mod(aid-1,length(corlist))+1};
            end
        else
            cor = corlist{mod(i-1,length(corlist))+1};
        end
        plot([cornerpoints(1,1),cornerpoints(2,1)],[cornerpoints(1,3),cornerpoints(2,3)],'Color',cor(1:3));
        hold on
        plot([cornerpoints(2,1),cornerpoints(4,1)],[cornerpoints(2,3),cornerpoints(4,3)],'Color',cor(1:3));
        plot([cornerpoints(4,1),cornerpoints(3,1)],[cornerpoints(4,3),cornerpoints(3,3)],'Color',cor(1:3));
        plot([cornerpoints(3,1),cornerpoints(1,1)],[cornerpoints(3,3),cornerpoints(1,3)],'Color',cor(1:3));
        id = strfind(p_data.labellist{i},'_');
        label = p_data.labellist{i};
        if(length(id)>0)
            label(id) = ' ';
        end
        min_x = min(cornerpoints(:,1));
        max_z = max(cornerpoints(:,3));
        t = text(min_x+0.01,max_z-0.02,[label,num2str(i)]);
        t.BackgroundColor = cor;
        t.FontSize = 8;
    else
        plot([cornerpoints(1,1),cornerpoints(2,1)],[cornerpoints(1,3),cornerpoints(2,3)],'k');
        hold on
        plot([cornerpoints(2,1),cornerpoints(4,1)],[cornerpoints(2,3),cornerpoints(4,3)],'k');
        plot([cornerpoints(4,1),cornerpoints(3,1)],[cornerpoints(4,3),cornerpoints(3,3)],'k');
        plot([cornerpoints(3,1),cornerpoints(1,1)],[cornerpoints(3,3),cornerpoints(1,3)],'k');
    end
end

axis equal

end

