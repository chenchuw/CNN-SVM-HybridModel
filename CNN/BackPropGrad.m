function [Wd_grad,bd_grad,Wc2_grad,bc2_grad,Wc1_grad,bc1_grad] = BackPropGrad(DeltaSoftmax,activationsPooled2, ...
    activationsPooled1,Wc2_grad,Wc1_grad,numFilters2,numFilters1,imageChannel,numImages,DeltaConv2,DeltaConv1,mb_images)

    % Compute gradients of dense layer
    Wd_grad = DeltaSoftmax*activationsPooled2';
    bd_grad = sum(DeltaSoftmax,2);
    
    % Compute gradients of second convolutional layer
    [Wc2_grad, bc2_grad] = Gradient(Wc2_grad, [numFilters2; numFilters1], numImages, DeltaConv2, activationsPooled1);
    
    % Compute gradients of first convolutional layer
    [Wc1_grad, bc1_grad] = Gradient(Wc1_grad, [numFilters1; imageChannel], numImages, DeltaConv1, mb_images);