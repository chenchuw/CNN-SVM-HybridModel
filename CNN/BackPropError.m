function [DeltaSoftmax,DeltaConv2,DeltaConv1] = BackPropError(probs,index,Wd,outputDim2,numFilters2, ...
    numImages,convDim2,poolDim2,activations2,outputDim1,numFilters1,Wc2,convDim1, ...
    poolDim1,activations1)

    % Error of dense layer
    output = zeros(size(probs));
    output(index) = 1;
    DeltaSoftmax = (probs - output);
    
    % Error of second pooling layer
    DeltaPool2 = reshape(Wd' * DeltaSoftmax,outputDim2,outputDim2,numFilters2,numImages);
    
    DeltaUnpool2 = zeros(convDim2,convDim2,numFilters2,numImages);        
    for imNum = 1:numImages
        for FilterNum = 1:numFilters2
            unpool = DeltaPool2(:,:,FilterNum,imNum);
            DeltaUnpool2(:,:,FilterNum,imNum) = kron(unpool,ones(poolDim2))./(poolDim2 ^ 2);
        end
    end 
    
    % Error of second convolutional layer
    DeltaConv2 = DeltaUnpool2 .* activations2 .* (1 - activations2);
    
    % Error of first pooling layer
    DeltaPooled1 = zeros(outputDim1,outputDim1,numFilters1,numImages);
    for i = 1:numImages
        for f1 = 1:numFilters1
            for f2 = 1:numFilters2
                DeltaPooled1(:,:,f1,i) = DeltaPooled1(:,:,f1,i) + convn(DeltaConv2(:,:,f2,i),Wc2(:,:,f1,f2),'full');
            end
        end
    end
    
    % Error of first convolutional layer   
    DeltaUnpool1 = zeros(convDim1,convDim1,numFilters1,numImages);
    for imNum = 1:numImages
        for FilterNum = 1:numFilters1
            unpool = DeltaPooled1(:,:,FilterNum,imNum);
            DeltaUnpool1(:,:,FilterNum,imNum) = kron(unpool,ones(poolDim1))./(poolDim1 ^ 2);
        end
    end
    
    DeltaConv1 = DeltaUnpool1 .* activations1 .* (1-activations1);

end