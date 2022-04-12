function data_split = svm_1v1(X, Y)
% split date into different classes

    classes=unique(Y);
    num_of_classes=length(classes);
    
    % split data by classes
    data_split=cell(num_of_classes,1);
    [mX,~] = size(X);
    
    for m=1:mX
        class=Y(m);
        data_split{class} = [data_split{class};X(m,:)];
    end
end
