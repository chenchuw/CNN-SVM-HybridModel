% Configuration
imageDim = 28;
numClasses = 10;  % Number of classes (MNIST images fall into 10 classes)
filterDim1 = 5;    % Filter size for conv layer
filterDim2 = 5;
imageChannel = 1;
numFilters1 = 10;
numFilters2 = 10;
poolDim1 = 2;
poolDim2 = 2;

%weight decay
lambda = 0.0001;

% Load MNIST Train and initialization
addpath ./common/;
images = loadMNISTImages('./common/train-images-idx3-ubyte');
images = reshape(images,imageDim,imageDim,1,[]);
labels = loadMNISTLabels('./common/train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

%only use 6000 images to speed up
%{
images = images(:,:,:,1:6000);
labels = labels(1:6000);
%}

Wc1 = 1e-1*randn(filterDim1,filterDim1,imageChannel,numFilters1);
Wc2 = 1e-1*randn(filterDim2,filterDim2,numFilters1,numFilters2);

outDim1 = imageDim - filterDim1 + 1; % dimension of convolved image
outDim1 = outDim1/poolDim1;
outDim2 = outDim1 - filterDim2 + 1; % dimension of convolved image
outDim2 = outDim2/poolDim2;
hiddenSize = outDim2^2*numFilters2;

r  = sqrt(6) / sqrt(numClasses+hiddenSize+1);
Wd = rand(numClasses, hiddenSize) * 2 * r - r;
%Wd(end, :) = 0;

bc1 = zeros(numFilters1, 1);
bc2 = zeros(numFilters2, 1);
bd = zeros(numClasses, 1);

epochs = 3;
minibatch = 150;
alpha = 1e-1;
momentum = .95;


m = length(labels); % training set size
% Setup for momentum
mom = 0.5;
momIncrease = 20;

Wc1_velocity = zeros(size(Wc1));
bc1_velocity = zeros(size(bc1));
Wc2_velocity = zeros(size(Wc2));
bc2_velocity = zeros(size(bc2));
Wd_velocity = zeros(size(Wd));
bd_velocity = zeros(size(bd));

it = 0;
C = [];
for e = 1:epochs
    
    % randomly permute indices of data for quick minibatch sampling
    rp = randperm(m);
    
    for s=1:minibatch:(m-minibatch+1)
        it = it + 1;

        % momentum enable
        if it == momIncrease
            mom = momentum;
        end

        % mini-batch pick
        mb_images = images(:,:,:,rp(s:s+minibatch-1));
        mb_labels = labels(rp(s:s+minibatch-1));
        
        numImages = size(mb_images,4);
        
        Wc2_grad = zeros(size(Wc2));
        Wc1_grad = zeros(size(Wc1));
        Wd_grad = zeros(size(Wd));
        bc2_grad = zeros(size(bc2));
        bc1_grad = zeros(size(bc1));
        bd_grad = zeros(size(bd));
        
        convDim1 = imageDim-filterDim1+1;
        outputDim1 = (convDim1)/poolDim1;
        convDim2 = outputDim1-filterDim2+1;
        outputDim2 = (convDim2)/poolDim2;
        
        %Feedfoward Pass
        activations1 = cnnConvolve4D(mb_images, Wc1, bc1);
        activationsPooled1 = cnnPool(poolDim1, activations1);
        activations2 = cnnConvolve4D(activationsPooled1, Wc2, bc2);
        activationsPooled2 = cnnPool(poolDim2, activations2);

        % Reshape activations into 2-d matrix, hiddenSize x numImages,
        % for Softmax layer
        activationsPooled2 = reshape(activationsPooled2,[],numImages);

        %% --------- Softmax Layer ---------
        [probs] = SoftMax(Wd, bd, activationsPooled2);

        %% --------- Calculate Cost ----------
        [cost, index] = CostFunction(probs, mb_labels, [[Wd(:)];[Wc1(:)];[Wc2(:)]], numImages,lambda);

        %% --------- Backpropagation ----------
        %errors
        % softmax layer
        output = zeros(size(probs));
        output(index) = 1;
        DeltaSoftmax = (probs - output);
        t = -DeltaSoftmax;
        
        % error of second pooling layer
        DeltaPool2 = reshape(Wd' * DeltaSoftmax,outputDim2,outputDim2,numFilters2,numImages);
        
        DeltaUnpool2 = zeros(convDim2,convDim2,numFilters2,numImages);        
        for imNum = 1:numImages
            for FilterNum = 1:numFilters2
                unpool = DeltaPool2(:,:,FilterNum,imNum);
                DeltaUnpool2(:,:,FilterNum,imNum) = kron(unpool,ones(poolDim2))./(poolDim2 ^ 2);
            end
        end 
        
        % error of second convolutional layer
        DeltaConv2 = DeltaUnpool2 .* activations2 .* (1 - activations2);
        
        %error of first pooling layer
        DeltaPooled1 = zeros(outputDim1,outputDim1,numFilters1,numImages);
        for i = 1:numImages
            for f1 = 1:numFilters1
                for f2 = 1:numFilters2
                    DeltaPooled1(:,:,f1,i) = DeltaPooled1(:,:,f1,i) + convn(DeltaConv2(:,:,f2,i),Wc2(:,:,f1,f2),'full');
                end
            end
        end
        
        %error of first convolutional layer   
        DeltaUnpool1 = zeros(convDim1,convDim1,numFilters1,numImages);
        for imNum = 1:numImages
            for FilterNum = 1:numFilters1
                unpool = DeltaPooled1(:,:,FilterNum,imNum);
                DeltaUnpool1(:,:,FilterNum,imNum) = kron(unpool,ones(poolDim1))./(poolDim1 ^ 2);
            end
        end
        
        DeltaConv1 = DeltaUnpool1 .* activations1 .* (1-activations1);
        
        %% ---------- Gradient Calculation ----------
        % softmax layer
        Wd_grad = DeltaSoftmax*activationsPooled2';
        bd_grad = sum(DeltaSoftmax,2);

        [Wc2_grad, bc2_grad] = Gradient(Wc2_grad, [numFilters2; numFilters1], numImages, DeltaConv2, activationsPooled1);

        
        % first convolutional layer
        [Wc1_grad, bc1_grad] = Gradient(Wc1_grad, [numFilters1; imageChannel], numImages, DeltaConv1, mb_images);
       
        
        
        
        Wd_velocity = mom*Wd_velocity + alpha*(Wd_grad/minibatch+lambda*Wd);
        bd_velocity = mom*bd_velocity + alpha*bd_grad/minibatch;
        Wc2_velocity = mom*Wc2_velocity + alpha*(Wc2_grad/minibatch+lambda*Wc2);
        bc2_velocity = mom*bc2_velocity + alpha*bc2_grad/minibatch;
        Wc1_velocity = mom*Wc1_velocity + alpha*(Wc1_grad/minibatch+lambda*Wc1);
        bc1_velocity = mom*bc1_velocity + alpha*bc1_grad/minibatch;
                        
        Wd = Wd - Wd_velocity;
        bd = bd - bd_velocity;
        Wc2 = Wc2 - Wc2_velocity;
        bc2 = bc2 - bc2_velocity;
        Wc1 = Wc1 - Wc1_velocity;
        bc1 = bc1 - bc1_velocity;
        
        fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
        C(length(C)+1) = cost;

    end
    
    alpha = alpha/2.0;
    
    testImages = loadMNISTImages('./common/t10k-images-idx3-ubyte');
    testImages = reshape(testImages,imageDim,imageDim,1,[]);
    testLabels = loadMNISTLabels('./common/t10k-labels-idx1-ubyte');
    testLabels(testLabels==0) = 10; % Remap 0 to 10
    
    %{
    testImages = testImages(:,:,:,1:1000);
    testLabels = testLabels(1:1000);
    %}
    
    activations1 = cnnConvolve4D(testImages, Wc1, bc1);
    activationsPooled1 = cnnPool(poolDim1, activations1);
    activations2 = cnnConvolve4D(activationsPooled1, Wc2, bc2);
    activationsPooled2 = cnnPool(poolDim2, activations2);

    % Reshape activations into 2-d matrix, hiddenSize x numImages,
    % for Softmax layer
    activationsPooled2 = reshape(activationsPooled2,[],length(testImages));


    probs = exp(bsxfun(@plus, Wd * activationsPooled2, bd));
    sumProbs = sum(probs, 1);
    probs = bsxfun(@times, probs, 1 ./ sumProbs);
    
    [~,preds] = max(probs,[],1);
    preds = preds';
    
    acc = sum(preds==testLabels)/length(preds);
    fprintf('Accuracy is %f\n',acc);
    plot(C);

end






