function W = train_svm_mhinge_sgd_mini(X,y,Delta,T,lambda,k,m_mini)
%

% relabel 0 to 10 for matlab
y(y == 0) = 10;

% dimension info
[m,d] = size(X);
% k = length(unique(y));    % not robust ;(
% k = 10;

% initialization
rng('default'); % ensure results consistancy during test
W0 = normrnd(1,0.1,d+1,k);
rng('default');
shuffle_index = randperm(m);
X = [ones(m,1) X];
X = X(shuffle_index,:);
y = y(shuffle_index);


W = W0;


for i = 1:round(T/(floor(m/m_mini)))
    %     j = rem(i,round(m/m_mini));
    
    %     if j == 0
    %         j = m;
    
    
    % reshuffle data for each epoch
    shuffle_index = randperm(m);
    X = X(shuffle_index,:);
    y = y(shuffle_index);
    %     end
    
    
    mini_batch_idx_mat = [1:floor(m/m_mini)]';
    mini_batch_idx_mat = repmat(mini_batch_idx_mat,1,m_mini);
    mini_batch_idx_mat = mini_batch_idx_mat(:);
    
    for g = 1:floor(m/m_mini)
        mini_batch_idx = find(mini_batch_idx_mat==g);
        subgradient_change = zeros(d+1,k);
        for j = 1:m_mini
            L_MH_mat = Delta(:,y(mini_batch_idx(j))) + (W-W(:,y(mini_batch_idx(j))))'*X(mini_batch_idx(j),:)';
            [~, yprime_index] = max(L_MH_mat);
            subgradient_change(:,yprime_index) = subgradient_change(:,yprime_index) + X(mini_batch_idx(j),:)';
            subgradient_change(:,y(mini_batch_idx(j))) = subgradient_change(:,y(mini_batch_idx(j))) - X(mini_batch_idx(j),:)';
        end
        subgradient = lambda*W + subgradient_change/m_mini;
        eta = 1/lambda/(i*m_mini-m_mini+g);
        W = W - eta*subgradient;
    end
    
    
    
    
end


end
