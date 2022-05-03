%% Configure CNN Structure
clear;
close all;

tic 

epochs = 3;
minibatch = 64;

imagedimension = 40;
numClasses = 3;  % Cell images has 3 classes
kerneldim1 = 5;    % Filter size for conv layer
kerneldim2 = 7;
Channel = 1;
numFilters1 = 20;
numFilters2 = 16;
poolbatch1 = 2;
poolbatch2 = 4;

% Weight decay
lambda = 0.0001;

Wc1 = 1e-1*randn(kerneldim1,kerneldim1,Channel,numFilters1);
Wc2 = 1e-1*randn(kerneldim2,kerneldim2,numFilters1,numFilters2);

outDim1 = imagedimension - kerneldim1 + 1; % dimension of convolved image
outDim1 = outDim1 / poolbatch1;
outDim2 = outDim1 - kerneldim2 + 1; % dimension of convolved image
outDim2 = outDim2 / poolbatch2;
hiddenSize = outDim2^2 * numFilters2;

r  = sqrt(6)/sqrt(numClasses+hiddenSize+1);
Wd = rand(numClasses, hiddenSize)*2*r-r;
bc1 = zeros(numFilters1, 1);
bc2 = zeros(numFilters2, 1);
bd = zeros(numClasses, 1);

% Setup for momentum
alpha = 1e-1;
momentum = .95;
mom = 0.5;
momIncrease = 20;

Wc1_velocity = zeros(size(Wc1));
bc1_velocity = zeros(size(bc1));
Wc2_velocity = zeros(size(Wc2));
bc2_velocity = zeros(size(bc2));
Wd_velocity = zeros(size(Wd));
bd_velocity = zeros(size(bd));

%% Load trainging data and initialization
%load MNIST-70k.mat;
load CellSet_raw_1600.mat;
% Reshape each image to 40*40 matrix
imagesTrain = reshape(imagesTrain,imagedimension,imagedimension,1,[]);
imagesTest = reshape(imagesTest,imagedimension,imagedimension,1,[]);

% Relabel label 0 to 10
labelsTrain(labelsTrain==0) = 10;
labelsTest(labelsTest==0) = 10;

m = length(labelsTrain);

%% Start running CNN
it = 0;
C = [];
for e = 1:epochs
    
    % Shuffle the data for quick minibatch sampling
    rp = randperm(m);
    
    for s=1:minibatch:(m-minibatch+1)
        it = it + 1;

        % Momentum enable
        if it == momIncrease
            mom = momentum;
        end

        %% Mini-batch pick
        mb_images = imagesTrain(:,:,:,rp(s:s+minibatch-1));
        mb_labels = labelsTrain(rp(s:s+minibatch-1));
        
        numImages = size(mb_images,4);
        
        Wc2_grad = zeros(size(Wc2));
        Wc1_grad = zeros(size(Wc1));
        
        convDim1 = imagedimension-kerneldim1+1;
        outputDim1 = (convDim1)/poolbatch1;
        convDim2 = outputDim1-kerneldim2+1;
        outputDim2 = (convDim2)/poolbatch2;
        
        %% --------- Feedforward Pass ---------
        activations1 = Convlayer(mb_images, Wc1, bc1);
        activationsPooled1 = Poolinglayer(poolbatch1, activations1);
        activations2 = Convlayer(activationsPooled1, Wc2, bc2);
        activationsPooled2 = Poolinglayer(poolbatch2, activations2);

        % Reshape activations into 2-d matrix, hiddenSize x numImages,
        % for Softmax layer
        activationsPooled2 = reshape(activationsPooled2,[],numImages);

        %% --------- Softmax Layer ---------
        [probs] = SoftMax(Wd, bd, activationsPooled2);

        %% --------- Calculate Cost ----------
        [cost, index] = CostFunction(probs, mb_labels, [[Wd(:)];[Wc1(:)];[Wc2(:)]], numImages,lambda);

        %% --------- Backpropagation ----------
        % Compute Error of each layer
        [DeltaSoftmax,DeltaConv2,DeltaConv1] = BackPropError(probs,index,Wd,outputDim2,numFilters2,numImages,convDim2, ...
            poolbatch2,activations2,outputDim1,numFilters1,Wc2,convDim1,poolbatch1,activations1);

        % Compute Gradient of each layer
        [Wd_grad,bd_grad,Wc2_grad,bc2_grad,Wc1_grad,bc1_grad] = BackPropGrad(DeltaSoftmax,activationsPooled2, ...
            activationsPooled1,Wc2_grad,Wc1_grad,numFilters2,numFilters1,Channel,numImages,DeltaConv2,DeltaConv1,mb_images);
        
        % Update gradients with momentum (W->W',b->b')
        [Wd,bd,Wc2,bc2,Wc1,bc1,Wd_velocity,bd_velocity,Wc2_velocity,bc2_velocity,Wc1_velocity,bc1_velocity] = GradUpdate( ...
            mom,alpha,minibatch,lambda,Wd_grad,bd_grad,Wc2_grad,bc2_grad,Wc1_grad,bc1_grad, ...
            Wd_velocity,bd_velocity,Wc2_velocity,bc2_velocity,Wc1_velocity,bc1_velocity, ...
            Wd,bd,Wc2,bc2,Wc1,bc1);
        
        %% Report cost on each iteration and maintain it
        %fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
        C(length(C)+1) = cost;

    end
    
    alpha = alpha/2.0;

end

%% Get features of train set using trained CNN model
activations1 = cnnConvolve4D(imagesTrain, Wc1, bc1);
activationsPooled1_train = CnnPooling(poolbatch1, activations1);
activations2 = cnnConvolve4D(activationsPooled1_train, Wc2, bc2);
activationsPooled2_train = CnnPooling(poolbatch2, activations2);

% Reshape features into 2d matrix (hiddenSize x numImages)
CNN_features_train = reshape(activationsPooled2_train,[],length(imagesTrain));

%% Get features of test set using trained CNN model
activations1 = cnnConvolve4D(imagesTest, Wc1, bc1);
activationsPooled1_test = CnnPooling(poolbatch1, activations1);
activations2 = cnnConvolve4D(activationsPooled1_test, Wc2, bc2);
activationsPooled2_test = CnnPooling(poolbatch2, activations2);

% Reshape features into 2d matrix (hiddenSize x numImages)
CNN_features_test = reshape(activationsPooled2_test,[],length(imagesTest));

%% Test features obtained from test set
% Compute classficication probabilities
probs = SoftMax(Wd,bd,CNN_features_test);
[~,preds] = max(probs,[],1);
preds = preds';

% Report the Accuracy
acc = sum(preds==labelsTest)/length(preds);
fprintf('CNN model Accuracy is %f\n',acc);
plot(C,'b');
title("CNN model - plot of Cost at each Iterations")
xlabel("Iterations")
ylabel("Cost")

%% Transpose CNN_features for SVM
CNN_features_train = CNN_features_train';
CNN_features_test = CNN_features_test';
fprintf("\nCNN features now ready for SVM... \n")

toc

