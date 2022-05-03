function [Wd,bd,Wc2,bc2,Wc1,bc1,Wd_velocity,bd_velocity,Wc2_velocity,bc2_velocity,Wc1_velocity,bc1_velocity] = GradUpdate(mom,alpha,minibatch,lambda, ...
    Wd_grad,bd_grad,Wc2_grad,bc2_grad,Wc1_grad,bc1_grad, ...
    Wd_velocity,bd_velocity,Wc2_velocity,bc2_velocity,Wc1_velocity,bc1_velocity, ...
    Wd,bd,Wc2,bc2,Wc1,bc1)

     % Update gradients with momentum (W->W',b->b')
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
end