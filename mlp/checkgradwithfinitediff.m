function grad = checkgradwithfinitediff(x,y,layernums,hidnums,outnums,active_fun,active_fun_grad)
% check the gradient computed by BP algorithm with finite difference
% ===initialize the weights of NN====
[D,N] = size(x);
unitnums = [D hidnums outnums];
for i=1:layernums,
    inputnums = unitnums(i) + 1;
    outputnums = unitnums(i+1);
    epsilon_init = sqrt(6)/sqrt(inputnums + outputnums);
    W{i} = rand(inputnums, outputnums)* 2 * epsilon_init - epsilon_init;
    W{i} = W{i}';
end

epsilon = 1e-6;

for l=1:layernums,
    [m,n] = size(W{l});
    for i=1:m,
        for j=1:n,
            W1 = W; W2 = W;
            W1{l}(i,j) = W{l}(i,j) + epsilon;
            W2{l}(i,j) = W{l}(i,j) - epsilon;
            % f(W1)
            rand('state',0);
            [~, ~, f1] = forwardprop(x,y,layernums,W1,active_fun);
            % f(W2)
            rand('state',0);
            [~, ~, f2] = forwardprop(x,y,layernums,W2,active_fun);
            % numerical gradient
            grad = (f1-f2)./(2*epsilon);
            [bp_grad, ~] = backprop(x,y,layernums,W,active_fun,active_fun_grad);
            bp_grad = bp_grad{l}(i,j);
            diff = abs(grad-bp_grad);
             fprintf('\nRelative Difference: %g\n', diff);
            assert(diff < 1e-7, 'layer: %d, unit: %d, unit: %d, the check failed\n',l,i,j);
%             if diff > 1e-6,
%                 fprintf('layer: %d, unit: %d, unit: %d, the check failed\n',l,i,j);
%             end
        end
    end
end
end