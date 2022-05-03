function W = train_svm_mhinge_sgd(X,y,Delta,T,lambda,k)
%
if nargin < 2
    Mat_file = 'mnist_LowDim.mat';
    Mat = load(Mat_file);
%     X = Mat.Xtr_LowDim;
%     y = Mat.ytr_LowDim;
    
    X = Mat.Xval_LowDim;
    y = Mat.yval_LowDim;
    
    T = 5e5;
    lambda = 1e-5;
end

% relabel 0 to 10 for matlab
y(y == 0) = 10;

% dimension info
[m,d] = size(X);
% k = length(unique(y));    % not robust ;(
% k = 10;

if nargin < 3
    Delta = ones(k,k) - eye(k,k);
end


% initialization
rng('default'); % ensure results consistancy during test
W0 = normrnd(1,0.1,d+1,k);
rng('default');
shuffle_index = randperm(m);
X = [ones(m,1) X];
X = X(shuffle_index,:);
y = y(shuffle_index);



W = W0;

for i = 1:T
    eta = 1/lambda/i;
    j = rem(i,m);
%     epoch = (i-j)/m;
    if j == 0
        j = m;
%         epoch = epoch + 1;
        
        % reshuffle data for each epoch
        shuffle_index = randperm(m);
        X = X(shuffle_index,:);
        y = y(shuffle_index);
    end
 
    L_MH_mat = Delta(:,y(j)) + (W-W(:,y(j)))'*X(j,:)';
    [~, yprime_index] = max(L_MH_mat);

    subgradient = lambda*W;
    subgradient(:,yprime_index) = subgradient(:,yprime_index) + X(j,:)';
    subgradient(:,y(j)) = subgradient(:,y(j)) - X(j,:)';
    
    W = W - eta*subgradient;
    
    
end


end
