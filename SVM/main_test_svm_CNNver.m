% This is the test code used for Multi-class SVM
close all;
% Plot default setting

set(0, 'DefaultTextFontSize', 18);
set(0, 'DefaultLineLineWidth', 2);
set(0, 'DefaultAxesLIneWidth', 2);
set(0, 'DefaultAxesFontSize', 14);
set(0, 'DefaultAxesFontName', 'Arial');


%% Data initialization/dimension reduction %%

cd('/Users/xiaoweige/Documents/Local Code/EC503-Final-Project/SVM');

%% Data loading
Mat_file = 'MNIST-70k.mat';
load(Mat_file);

% transpose once from original set once
% m = matfile(Mat_file,'Writable',true); % load as handle of .mat
% m.imagesTest = imagesTest';
% m.imagesTrain = imagesTrain';
% m.labelsTrain = labelsTrain;
% m.labelsTest = labelsTest;

Xtr = imagesTrain;
Xte = imagesTest;
ytr = labelsTrain;
yte = labelsTest;

%% Reshape data trial with 1 sample
close all;

% i = randi(size(Xtr,1));
% figure;     % vis a ramdom sample
% imagesc(permute(reshape(Xtr(i,:),28,28), [2 1]));   % permute and transpose for display purpose only
% axis(gca,'equal','tight');

Xtr_reshaped = permute(reshape(Xtr(i,:),28,28)', [2 1]);
Xtr_reshaped_LowDim = imresize(full(Xtr_reshaped),12/28);

pause(0.2);
figure;
subplot(1,2,1);
imagesc(permute(reshape(Xtr(i,:),28,28)', [2 1]));   % permute and transpose for display purpose only
axis(gca,'equal','tight');
subplot(1,2,2);
imagesc(Xtr_reshaped_LowDim);   
axis(gca,'equal','tight');
title(['The number is ', num2str(ytr(i))]);


%% Reshape data trial with 1 sample (test)
close all;

i = randi(size(Xte,1));
% figure;     % vis a ramdom sample
% imagesc(permute(reshape(Xtr(i,:),28,28), [2 1]));   % permute and transpose for display purpose only
% axis(gca,'equal','tight');

Xte_reshaped = permute(reshape(Xte(i,:),28,28), [2 1]);
Xte_reshaped_LowDim = imresize(full(Xte_reshaped),16/28);

pause(0.2);
figure;
subplot(1,2,1);
imagesc(permute(reshape(Xte(i,:),28,28)', [2 1]));   % permute and transpose for display purpose only
axis(gca,'equal','tight');
subplot(1,2,2);
imagesc(Xte_reshaped_LowDim');   
axis(gca,'equal','tight');
title(['The number is ', num2str(yte(i))]);

%% reshape all dataset and save once
% Mat_file = 'mnist-70k_LowDim_4.mat';
% m = matfile(Mat_file,'Writable',true); % load as handle of .mat
% % dim_reduce_ratio = 10/28;
% dim_pixel_number = 2;
% dim_reduce_ratio = dim_pixel_number/28;
% 
% X = full(Xtr);
% data_number = size(X,1);
% X_reshaped = reshape(X,data_number,28,28);
% X_reshaped_LowDim = imresize(permute(X_reshaped,[2,3,1]),dim_reduce_ratio);
% X_LowDim = sparse(reshape(permute(X_reshaped_LowDim,[3,1,2]),data_number,dim_pixel_number^2));
% m.Xtr_LowDim = X_LowDim;
% ytr(ytr==0) = 10;
% m.ytr_LowDim = ytr;
% 
% X = full(Xte);
% data_number = size(X,1);
% X_reshaped = reshape(X,data_number,28,28);
% X_reshaped_LowDim = imresize(permute(X_reshaped,[2,3,1]),dim_reduce_ratio);
% X_LowDim = sparse(reshape(permute(X_reshaped_LowDim,[3,1,2]),data_number,dim_pixel_number^2));
% m.Xte_LowDim = X_LowDim;
% yte(yte==0) = 10;
% m.yte_LowDim = yte;

%% Random low dimension data generating and save once

% rng('default'); % for reproducibility
% 
% rand_feature_idx = randi(28*28,256,1);
% 
% Mat_file = 'mnist-70k_LowDim_rand.mat';
% m = matfile(Mat_file,'Writable',true); % load as handle of .mat
% 
% m.Xtr_LowDim_rand = Xtr(:,rand_feature_idx);
% m.Xte_LowDim_rand = Xte(:,rand_feature_idx);
% m.ytr_LowDim_rand = ytr;
% m.yte_LowDim_rand = yte;

%% Model training with SGD multiclass SVM %%
clear
close all
clc

%% load dataset
% % Raw data set SVM test
% load('MNIST-70K.mat');
% 
% Xtr_LowDim = imagesTrain;
% ytr_LowDim = labelsTrain;
% ytr_LowDim(ytr_LowDim==0)=10;
% [m,d] = size(Xtr_LowDim);
% Xval_LowDim = Xtr_LowDim(1:round(m/6),:);
% yval_LowDim = ytr_LowDim(1:round(m/6),:);
% Xtr_LowDim = Xtr_LowDim((round(m/6)+1):end,:);
% ytr_LowDim = ytr_LowDim((round(m/6)+1):end,:);
% Xte_LowDim = imagesTest;
% yte_LowDim = labelsTest;
% yte_LowDim(yte_LowDim==0)=10;


% % 256 features selected by Bicubic interpolation: weighted average of 4-by-4 pixels
% % load('mnist-70k_LowDim.mat')
% load('mnist-70k_LowDim_4.mat')
% [m,d] = size(Xtr_LowDim);
% Xval_LowDim = Xtr_LowDim(1:round(m/6),:);
% yval_LowDim = ytr_LowDim(1:round(m/6),:);
% Xtr_LowDim = Xtr_LowDim((round(m/6)+1):end,:);
% ytr_LowDim = ytr_LowDim((round(m/6)+1):end,:);
% Xte_LowDim = Xte_LowDim;
% yte_LowDim = yte_LowDim;


% Random 100 features test
load('mnist-70k_LowDim_rand.mat')
[m,d] = size(Xtr_LowDim);
Xval_LowDim = Xtr_LowDim_rand(1:round(m/6),:);
yval_LowDim = ytr_LowDim_rand(1:round(m/6),:);
Xtr_LowDim = Xtr_LowDim_rand((round(m/6)+1):end,:);
ytr_LowDim = ytr_LowDim_rand((round(m/6)+1):end,:);
Xte_LowDim = Xte_LowDim_rand;
yte_LowDim = yte_LowDim_rand;


% % CNN-SVM connection initial test
% load('CNN_256features.mat');
% Xtr_LowDim = CNN_features_train;
% ytr_LowDim = labelsTrain;
% [m,d] = size(Xtr_LowDim);
% Xval_LowDim = Xtr_LowDim(1:round(m/6),:);
% yval_LowDim = ytr_LowDim(1:round(m/6),:);
% Xtr_LowDim = Xtr_LowDim((round(m/6)+1):end,:);
% ytr_LowDim = ytr_LowDim((round(m/6)+1):end,:);
% Xte_LowDim = CNN_features_test;
% yte_LowDim = labelsTest;


%% Other tests

% Set delta
% k = length(unique(ytr));  % not robust
k = 10;
Delta = ones(k,k) - eye(k,k);

% % Different penalty test
% load('Conf_mat.mat');
% Delta = Delta + Delta.*Conf_mat_LowDim;


%% Select lambda using the validation set
T = 1e5;    % set training total iterations

lambda = 10.^[-5:0.3:-1];
val_error = zeros(length(lambda),1);
for i = 1:length(lambda)
    tic 
    W = train_svm_mhinge_sgd(Xtr_LowDim,ytr_LowDim,Delta,T,lambda(i));
    toc
    ypred = test_svm_multi(W, Xval_LowDim);
    val_error(i) = mean(ypred~=yval_LowDim);
end

val_error
figure;
semilogx(lambda,val_error);
title('Validation error with different \lambda');
xlabel('\lambda');
ylabel('Error');

[~,lambda_opt_index] = min(val_error);
fprintf('Best lambda is: %.2e\n',lambda(lambda_opt_index));

%% Train with selected lambda on training+validation sets

T = 1e6;    % adjust training iterations if needed

lambda_opt = lambda(lambda_opt_index);
% lambda_opt = 1e-3;
tic
W = train_svm_mhinge_sgd([Xtr_LowDim;Xval_LowDim],[ytr_LowDim;yval_LowDim],Delta,T,lambda_opt);
toc

% Test
ypred = test_svm_multi(W, Xte_LowDim);
test_accuracy = mean(ypred==yte_LowDim);
fprintf('Test accuracy is %.2f%%\n',test_accuracy*100);

%% Compute confusion matrix
ConfMat = zeros(k,k);

for i = 1:length(yte_LowDim)
    ConfMat(ypred(i),yte_LowDim(i)) = ConfMat(ypred(i),yte_LowDim(i))+1;
end

figure;
% colormap(cmap)
% set(gca,'XTick',[],'YTick',[],'YDir','normal')
imagesc(ConfMat);
[x,y] = meshgrid(1:k,1:k);
text(x(:),y(:),num2str(ConfMat(:)),'HorizontalAlignment','center','Color','r');
axis(gca,'equal','tight');

% save confusion mat for different delta once
Mat_file = 'Conf_mat.mat';
m = matfile(Mat_file,'Writable',true); % load as handle of .mat
m.Conf_mat_LowDim = ConfMat;