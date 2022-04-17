function [probs] = SoftMax(Weight, Bias, Layer)
    probs = exp(bsxfun(@plus, Weight * Layer, Bias));
    sumProbs = sum(probs, 1);
    probs = bsxfun(@times, probs, 1 ./ sumProbs);
end