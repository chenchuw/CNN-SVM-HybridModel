function y = test_svm_multi(W, X)

    
    m = size(X,1);
    X = [ones(m,1) X];
    [~,y] = max(X*W,[],2);
    
end