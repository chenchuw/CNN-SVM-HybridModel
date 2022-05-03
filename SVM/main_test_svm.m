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

% transpose once from original set
m = matfile(Mat_file,'Writable',true); % load as handle of .mat
m.Xtr = Xtr';
m.Xval = Xval';
m.Xte = Xte';
m.ytr = ytr';
m.yval = yval';
m.yte = yte';

%% Reshape data trial with 1 sample
close all;

i = randi(size(Xtr,1));
figure;     % vis a ramdom sample
imagesc(permute(reshape(Xtr(i,:),28,28), [2 1]));   % permute for display purpose only
axis(gca,'equal','tight');

Xtr_reshaped = permute(reshape(Xtr(i,:),28,28), [2 1]);
Xtr_reshaped_LowDim = imresize(full(Xtr_reshaped),1/2.8);

pause(0.2);
figure;
imagesc(Xtr_reshaped_LowDim);   
axis(gca,'equal','tight');

%% reshape all dataset and save once
% Mat_file = 'mnist_LowDim.mat';
% m = matfile(Mat_file,'Writable',true); % load as handle of .mat
% dim_reduce_ratio = 1/2.8;
% 
% X = full(Xtr);
% data_number = size(X,1);
% X_reshaped = reshape(X,data_number,28,28);
% X_reshaped_LowDim = imresize(permute(X_reshaped,[2,3,1]),dim_reduce_ratio);
% X_LowDim = sparse(reshape(permute(X_reshaped_LowDim,[3,1,2]),data_number,10*10));
% m.Xtr_LowDim = X_LowDim;
% m.ytr_LowDim = ytr;
% 
% X = full(Xval);
% data_number = size(X,1);
% X_reshaped = reshape(X,data_number,28,28);
% X_reshaped_LowDim = imresize(permute(X_reshaped,[2,3,1]),dim_reduce_ratio);
% X_LowDim = sparse(reshape(permute(X_reshaped_LowDim,[3,1,2]),data_number,10*10));
% m.Xval_LowDim = X_LowDim;
% m.yval_LowDim = yval;
% 
% X = full(Xte);
% data_number = size(X,1);
% X_reshaped = reshape(X,data_number,28,28);
% X_reshaped_LowDim = imresize(permute(X_reshaped,[2,3,1]),dim_reduce_ratio);
% X_LowDim = sparse(reshape(permute(X_reshaped_LowDim,[3,1,2]),data_number,10*10));
% m.Xte_LowDim = X_LowDim;
% m.yte_LowDim = yte;

%% Random low dimension data generating and save once

% rng('default'); % for reproducibility
% 
% rand_feature_idx = randi(28*28,100,1);
% 
% Mat_file = 'mnist_LowDim_rand.mat';
% m = matfile(Mat_file,'Writable',true); % load as handle of .mat
% 
% m.Xtr_LowDim = Xtr(:,rand_feature_idx);
% m.Xval_LowDim = Xval(:,rand_feature_idx);
% m.Xte_LowDim = Xte(:,rand_feature_idx);
% m.ytr_LowDim = ytr;
% m.yte_LowDim = yte;
% m.yval_LowDim = yval;

%% Model training with SGD multiclass SVM %%
clear
close all
clc

%% load dataset
% Raw data set SVM test
load('MNIST-70K.mat');
Xtr_LowDim = imagesTrain';
ytr_LowDim = labelsTrain;
[m,d] = size(Xtr_LowDim);
Xval_LowDim = Xtr_LowDim(1:round(m/6),:);
yval_LowDim = ytr_LowDim(1:round(m/6),:);
Xtr_LowDim = Xtr_LowDim((round(m/6)+1):end,:);
ytr_LowDim = ytr_LowDim((round(m/6)+1):end,:);
Xte_LowDim = imagesTest';
yte_LowDim = labelsTest;


% load('mnist_LowDim.mat')

% Random 100 features test
% load('mnist_LowDim_rand.mat')

% % CNN-SVM connection initial test
% load('CNN_features.mat');
% Xtr_LowDim = CNN_features_train;
% ytr_LowDim = trainLabels;
% [m,d] = size(Xtr_LowDim);
% Xval_LowDim = Xtr_LowDim(1:round(m/6),:);
% yval_LowDim = ytr_LowDim(1:round(m/6),:);
% Xtr_LowDim = Xtr_LowDim((round(m/6)+1):end,:);
% ytr_LowDim = ytr_LowDim((round(m/6)+1):end,:);
% Xte_LowDim = CNN_features_test;
% yte_LowDim = testLabels;


% Set delta
% k = length(unique(ytr));  % not robust
k = 10;
Delta = ones(k,k) - eye(k,k);

% % Different penalty test
% load('Conf_mat.mat');
% Delta = Delta + Delta.*Conf_mat_LowDim;


%% Select lambda using the validation set
T = 1e5;    % set training total iterations

lambda = 10.^[-3:2:10];
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

T = 1e7;    % adjust training iterations if needed

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