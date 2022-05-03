function PooledFeatures = Poolinglayer(poolDim, convolvedFeatures)
% CnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

PooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

    for imageNum = 1:numImages
        for featureNum = 1:numFilters
            featuremap = squeeze(convolvedFeatures(:,:,featureNum,imageNum));
            pooledFeaturemap = conv2(featuremap,ones(poolDim)/(poolDim^2),'valid');
            PooledFeatures(:,:,featureNum,imageNum) = pooledFeaturemap(1:poolDim:end,1:poolDim:end);
        end
    end
end

