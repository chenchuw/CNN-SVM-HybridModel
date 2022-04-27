function segment_label2individual(img_name,FOV,filepath_save)
close all;

% cd(filepath);

t = Tiff(img_name);
img_float32 = read(t);
% rescale to 0~1
img_float32 = (img_float32-min(img_float32,[],'all'))/(max(img_float32,[],'all')-min(img_float32,[],'all'));

t = Tiff(append(img_name,'f'));
img_int16 = read(t);

img_size = size(img_int16,1);
img_resize = 40;
cell_number = max(img_int16,[],'all');

% single_cell_img_mat = zeros(img_resize,img_resize,cell_number);
% fprintf('Total %d\n', cell_number);
% over = 0;

for i = 1:cell_number
    temp = zeros(img_size,img_size);
    temp(img_int16==i) = img_float32(img_int16==i);
    
    measurements = regionprops(temp~=0, 'BoundingBox');
    croppedImage = imcrop(temp, measurements.BoundingBox);
    
%     figure(1);
%     imshow(croppedImage);
    [a,b] = size(croppedImage);
    if a >= 40 || b>=40
%         fprintf('Too large');
%         over = over+1;
       continue;
    end
%     pause(0.5);
    
    K_pad = padarray(croppedImage, [floor((img_resize-a)/2) floor((img_resize-b)/2)], 0,'post');
    K_pad = padarray(K_pad, [ceil((img_resize-a)/2) ceil((img_resize-b)/2)], 0,'pre');
    
    TableName = append(filepath_save,'FOV',num2str(FOV),'_Cell',num2str(i),'.csv');
%     writematrix(K_pad,TableName);
    
    
%     imshow(K_pad);
 
end

% fprintf('Oversize %d\n', over);

    
    

end