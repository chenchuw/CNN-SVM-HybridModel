function [Cost, index] = CostFunction(probs, Label, Weight, NumImages,lambda)
val = 0;    
logp = log(probs);
    index = sub2ind(size(logp),Label',1:size(probs,2));
    ceCost = -sum(logp(index)); 
    for i = 1:length(Weight)
        val = val + sum(Weight(i).^2);
    end
    wCost = lambda/2 * val;
    Cost = ceCost/NumImages + wCost;
end