
%% Segmenting to single cell and zero padding
filepath = '/Users/xiaoweige/Documents/Local Data/20220404-05 EC503_CHshape/BT5_Single/';

keyword = '*.tif';
listing = dir([filepath, keyword]);
name = string(extractfield(listing, 'name'));
fprintf('Data in file:\n');
name_seq = [[1:length(name)]', name'];
disp(name_seq);

for i = 1:length(name)
    segment_label2individual(append(filepath, name(i)),i,[filepath,'BT5_SingleCell/'])   
end


%% Data augmentation of CS and BP

filepath = '/Users/xiaoweige/Documents/Local Data/20220404-05 EC503_CHshape/CS5_Single/CS5_SingleCell/';

keyword = '*.csv';
listing = dir([filepath, keyword]);
name = string(extractfield(listing, 'name'));
% fprintf('Data in file:\n');
% name_seq = [[1:length(name)]', name'];
% disp(name_seq);

for i = 1:length(name)
    T = readmatrix(append(filepath,name(i)));
    for j = 1:20
        T_rotate = imrotate(T,j*18,'crop');
        TableName = append(filepath,name(i),'rotate',num2str(j*18),'.csv');
%         writematrix(T_rotate,TableName);
    end
    
end

%% Reshape image to one matrix

filepath = '/Users/xiaoweige/Documents/Local Data/20220404-05 EC503_CHshape/BT5_Single/BT5_SingleCell/';

keyword = '*.csv';
listing = dir([filepath, keyword]);
name = string(extractfield(listing, 'name'));
% fprintf('Data in file:\n');
% name_seq = [[1:length(name)]', name'];
% disp(name_seq);
T_whole = zeros(length(name),40*40);

for i = 1:length(name)
    T = readmatrix(append(filepath,name(i)));
    T_whole(i,:) = T(:);
end

TableName = 'BT5_whole.csv';
writematrix(T_whole,TableName);

%% Recombine for training and testing set
% BP: label 1, BT: label 2, CS: label 3

T_BP = readmatrix('BP5_whole.csv');
T_BT = readmatrix('BT5_whole.csv');
T_CS = readmatrix('CS5_whole.csv');

T_label = [ones(size(T_BP,1),1); ones(size(T_BT,1),1)*2; ones(size(T_CS,1),1)*3];

T_set = [T_BP;T_BT;T_CS];

rng('default');
sample_number = size(T_set,1);
rand_indices = randperm(sample_number);
Test_indices = rand_indices(1:round(sample_number/7));
Train_indices = rand_indices((round(sample_number/7)+1):end);

imagesTrain = T_set(Train_indices,:);
imagesTest = T_set(Test_indices,:);
labelsTrain = T_label(Train_indices,:);
labelsTest = T_label(Test_indices,:);

Mat_file = 'CellSet.mat';
m = matfile(Mat_file,'Writable',true); 
% load as handle of .mat
m.imagesTrain = imagesTrain;
m.imagesTest = imagesTest;
m.labelsTrain = labelsTrain;
m.labelsTest = labelsTest;


