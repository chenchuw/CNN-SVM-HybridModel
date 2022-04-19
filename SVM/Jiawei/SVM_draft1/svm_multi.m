function SVMModels = svm_multi(X, Y)
% OvO multiclass SVM: return k(k-1)/2 predictors

    classes=unique(Y);
    num_of_classes=length(classes);

    % create OvO SVM multiclassifier
    num_of_models=num_of_classes*(num_of_classes-1)/2;
    SVMModels=cell(num_of_models,2);
    
    pred_no = 1;
    data_split = svm_1v1(X, Y);

    for i = 1:num_of_models
        for j= i+1:num_of_models
            SVMModels{pred_no,1}=[i,j];
            % OvO
            Xt = [data_split{i}; data_split{j}];
            [mi,~]=size(data_split{i});
            [mj,~]=size(data_split{j});
            yt = [ones(mi,1);zeros(mj,1)];
            %predictor
            SVMModels{pred_no,2}=fitcsvm(Xt,yt,'ClassNames',[false true],'Standardize',true,...
                'KernelFunction','RBF');
            
            pred_no = pred_no + 1;
        end
    end
    
end
