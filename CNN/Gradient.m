function [grad, bc_grad] = Gradient(grad, Filters, NumImages, Error, Feature_Map)
    for fil2 = 1:Filters(1)
        for fil1 = 1:Filters(2)
            for im = 1:NumImages
                    grad(:,:,fil1,fil2) = grad(:,:,fil1,fil2) + conv2(Feature_Map(:,:,fil1,im),rot90(Error(:,:,fil2,im),2),'valid');
            end
        end
        temp = Error(:,:,fil2,:);
        bc_grad(fil2) = sum(temp(:));
    end
end