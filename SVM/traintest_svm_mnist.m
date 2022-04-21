clear
close all
clc

%% load dataset
load('mnist_LowDim.mat')

% Set T
T = 1e5;
% Set delta
% k = length(unique(ytr));  % not robust
k = 10;
Delta = ones(k,k) - eye(k,k);


%% Select lambda using the validation set
lambda = 10.^[18:0.2:19.2];
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

%% Train with selected lambda on training+validation sets

W = train_svm_mhinge_sgd([Xtr_LowDim;Xval_LowDim],[ytr_LowDim;yval_LowDim],Delta,T,lambda(lambda_opt_index));

% Test
ypred = test_svm_multi(W, Xte_LowDim);
test_err = mean(ypred~=yte_LowDim);
fprintf('Test error is %f%%\n',test_err*100);

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

%% Plot default

% set(0, 'DefaultTextFontSize', 18);
% set(0, 'DefaultLineLineWidth', 2);
% set(0, 'DefaultAxesLIneWidth', 2);
% set(0, 'DefaultAxesFontSize', 14);
% set(0, 'DefaultAxesFontName', 'Arial');