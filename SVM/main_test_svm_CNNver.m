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
Mat_file = 'CellSet_raw_1600.mat';
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

% %% reshape all dataset and save once
% Mat_file = 'CellSet_raw_LowDim_400.mat';
% m = matfile(Mat_file,'Writable',true); % load as handle of .mat
% % dim_reduce_ratio = 10/28;
% img_size = 40;
% dim_pixel_number = 20;
% dim_reduce_ratio = dim_pixel_number/img_size;
% 
% X = full(Xtr);
% data_number = size(X,1);
% X_reshaped = reshape(X,data_number,img_size,img_size);
% X_reshaped_LowDim = imresize(permute(X_reshaped,[2,3,1]),dim_reduce_ratio);
% X_LowDim = (reshape(permute(X_reshaped_LowDim,[3,1,2]),data_number,dim_pixel_number^2));
% m.Xtr_LowDim = X_LowDim;
% ytr(ytr==0) = 10;
% m.ytr_LowDim = ytr;
% 
% X = full(Xte);
% data_number = size(X,1);
% X_reshaped = reshape(X,data_number,img_size,img_size);
% X_reshaped_LowDim = imresize(permute(X_reshaped,[2,3,1]),dim_reduce_ratio);
% X_LowDim = (reshape(permute(X_reshaped_LowDim,[3,1,2]),data_number,dim_pixel_number^2));
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
% Raw data set SVM test
load('MNIST-70K.mat');

Xtr_LowDim = imagesTrain;
ytr_LowDim = labelsTrain;
ytr_LowDim(ytr_LowDim==0)=10;
[m,d] = size(Xtr_LowDim);
Xval_LowDim = Xtr_LowDim(1:round(m/6),:);
yval_LowDim = ytr_LowDim(1:round(m/6),:);
Xtr_LowDim = Xtr_LowDim((round(m/6)+1):end,:);
ytr_LowDim = ytr_LowDim((round(m/6)+1):end,:);
Xte_LowDim = imagesTest;
yte_LowDim = labelsTest;
yte_LowDim(yte_LowDim==0)=10;


% 256 features selected by Bicubic interpolation: weighted average of 4-by-4 pixels
% % load('mnist-70k_LowDim.mat')
% load('mnist-70k_LowDim.mat')
% [m,d] = size(Xtr_LowDim);
% Xval_LowDim = full(Xtr_LowDim(1:round(m/6),:));
% yval_LowDim = full(ytr_LowDim(1:round(m/6),:));
% Xtr_LowDim = full(Xtr_LowDim((round(m/6)+1):end,:));
% ytr_LowDim = full(ytr_LowDim((round(m/6)+1):end,:));
% Xte_LowDim = full(Xte_LowDim);
% yte_LowDim = full(yte_LowDim);


% % Random 100 features test
% load('mnist-70k_LowDim_rand.mat')
% [m,d] = size(Xtr_LowDim);
% Xval_LowDim = Xtr_LowDim_rand(1:round(m/6),:);
% yval_LowDim = ytr_LowDim_rand(1:round(m/6),:);
% Xtr_LowDim = Xtr_LowDim_rand((round(m/6)+1):end,:);
% ytr_LowDim = ytr_LowDim_rand((round(m/6)+1):end,:);
% Xte_LowDim = Xte_LowDim_rand;
% yte_LowDim = yte_LowDim_rand;


% % CNN-SVM connection initial test
% load('CNN_256.mat');
% Xtr_LowDim = CNN_features_train;
% ytr_LowDim = labelsTrain;
% [m,d] = size(Xtr_LowDim);
% Xval_LowDim = Xtr_LowDim(1:round(m/6),:);
% yval_LowDim = ytr_LowDim(1:round(m/6),:);
% Xtr_LowDim = Xtr_LowDim((round(m/6)+1):end,:);
% ytr_LowDim = ytr_LowDim((round(m/6)+1):end,:);
% Xte_LowDim = CNN_features_test;
% yte_LowDim = labelsTest;


%% load cell dataset

% Raw data SVM test
load('CellSet_raw_1600.mat');
Xtr_LowDim = full(imagesTrain);
ytr_LowDim = full(labelsTrain);
[m,d] = size(Xtr_LowDim);
Xval_LowDim = Xtr_LowDim(1:round(m/6),:);
yval_LowDim = ytr_LowDim(1:round(m/6),:);
Xtr_LowDim = Xtr_LowDim((round(m/6)+1):end,:);
ytr_LowDim = ytr_LowDim((round(m/6)+1):end,:);
Xte_LowDim = full(imagesTest);
yte_LowDim = full(labelsTest);

% 
% % Low Dim sub sampling data test
% load('CellSet_raw_LowDim_4.mat');
% [m,d] = size(Xtr_LowDim);
% Xval_LowDim = Xtr_LowDim(1:round(m/6),:);
% yval_LowDim = ytr_LowDim(1:round(m/6),:);
% Xtr_LowDim = Xtr_LowDim((round(m/6)+1):end,:);
% ytr_LowDim = ytr_LowDim((round(m/6)+1):end,:);
% Xte_LowDim = Xte_LowDim;
% yte_LowDim = yte_LowDim;


% CNN-SVM connection initial test
% load('Cellset_CNN_256_v3.mat');
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
T_SVM = 1e5;    % set training total iterations
m_mini = 4;
lambda = 10.^[-6:1:0];
val_error = zeros(length(lambda),1);
for i = 1:length(lambda)
    tic 
%     W = train_svm_mhinge_sgd(Xtr_LowDim,ytr_LowDim,Delta,T_SVM,lambda(i),k);
    W = train_svm_mhinge_sgd_mini(Xtr_LowDim,ytr_LowDim,Delta,T_SVM,lambda(i),k,m_mini);
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

T_SVM = 1e6;    % adjust training iterations if needed

lambda_opt = lambda(lambda_opt_index);
lambda_opt = 1e-2;
tic
% W = train_svm_mhinge_sgd([Xtr_LowDim;Xval_LowDim],[ytr_LowDim;yval_LowDim],Delta,T_SVM,lambda_opt,k);
W = train_svm_mhinge_sgd_mini([Xtr_LowDim;Xval_LowDim],[ytr_LowDim;yval_LowDim],Delta,T_SVM,lambda_opt,k,m_mini);
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


%% plot accuracy MNIST
figure;
T_SVM =[92.1500;92.2300;92.2600;92.1600;91.0100;80.6900;28.9800];
FeatureNumber = [784;400;256;144;64;16;4];
loglog(FeatureNumber,T_SVM,'*-')
set(gca, 'XTick', flip(FeatureNumber));
grid on;
xlabel('Number of features');
ylabel('Accuracy (%)');
xlim([3 900]);
ylim([28 95]);


%%
figure;
T_CNN =[96.41;95.98;94.45;86.71];
T_SVM =[92.2600;92.1600;91.0100;80.6900];
T_CNN_SVM = [97.7;97.78;96.17;86.91];
FeatureNumber = [256;144;64;16];
loglog(FeatureNumber,T_SVM,'*-');
hold on;
loglog(FeatureNumber,T_CNN,'o-');
grid on;
loglog(FeatureNumber,T_CNN_SVM,'x-');
grid on;
xlabel('Number of features');
ylabel('Accuracy (%)');
xlim([13 300]);
ylim([80 100]);
legend('SVM','CNN','CNN-SVM');
legend box off;
set(gca, 'XTick', flip(FeatureNumber));

%% plot accuracy Cell
figure;
T_SVM =[82.61;80.91;78.94;70.65;69.5;64.18;52.06];
FeatureNumber = [1600;400;256;144;64;16;4];
loglog(FeatureNumber,T_SVM,'*-')
set(gca, 'XTick', flip(FeatureNumber));
% grid on;
xlabel('Number of features');
ylabel('Accuracy (%)');
xlim([3 1800]);
ylim([50 95]);
legend('SVM with down sampling features');
legend box off
%%
figure;
T_SVM =[82.61;78.94;70.65];
T_CNN =[84.64;84.26];
T_CNN_SVM = [87.00;85.35];
FeatureNumber1 = [1600;256;144];
FeatureNumber2 = [256;144];
loglog(FeatureNumber1,T_SVM,'*-');
hold on;
loglog(FeatureNumber2,T_CNN,'o-');
grid on;
loglog(FeatureNumber2,T_CNN_SVM,'x-');
grid on;
xlabel('Number of features');
ylabel('Accuracy (%)');
xlim([100 1800]);
ylim([70 90]);
legend('SVM','CNN','CNN-SVM');
legend box off;
set(gca, 'XTick', flip(FeatureNumber));