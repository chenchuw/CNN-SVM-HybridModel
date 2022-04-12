function [y_pred]=svm_predict(SVMModels, t)
% get prediction of t dataset

    num_of_models=length(SVMModels);
    [mt,~]=size(t);
    labels=zeros(mt,num_of_models); % prediction result of each predictor
    preds =zeros(mt,num_of_models); % transfer 0/1 to Y classes

    for i=1:num_of_models
        [label,~]=predict(SVMModels{i,2},t);
        labels(:,i)=label;
        % transfer 0/1 to Y classes
        for j=1:mt
            if label(j)==1
                preds(j,i)=SVMModels{i,1}(1);
            else
                preds(j,i)=SVMModels{i,1}(2);
            end
        end
    end
    
    % get the prediction which has the highest number if wins
    y_pred = zeros(mt,1);
    for yi=1:mt
        y_pred(yi)=mode(preds(yi,:));
    end

end