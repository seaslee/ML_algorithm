function [W_grad, cost] = backprop(x,y,layernums,W,active_fun,active_fun_grad,varargin)
% backpropagation algo to computer gradient

s.dropout = 0;
s.visualdroprate = 0.5;
s.hiddendroprate = 0.5;
if ~isempty(varargin),
    s = handleoptions(s,varargin);
end

dropout = s.dropout;
visualdroprate = s.visualdroprate;
hiddendroprate = s.hiddendroprate;

n = size(x, 2);
%====forward-propagation to computer value of every node in the net
if dropout,
    [z, a, cost, rmask] = forwardprop(x,y,layernums,W,active_fun,...
                                      'dropout',dropout,...
                                      'visualdroprate',visualdroprate,...
                                      'hiddendroprate',hiddendroprate);
else
    [z, a, cost, rmask] = forwardprop(x,y,layernums,W,active_fun);
end

%computer the delta value for every node in the net
for i=layernums+1:-1:2,
    if i==layernums+1,
        delta{i}=(a{i}-y);
    else
        tmp = delta{i+1}'*W{i};
        tmp = tmp(:,2:end)';
        delta{i} = tmp.*active_fun_grad(z{i});
        %dropout
        if dropout,
%             fprintf('funck bp');
            delta{i} = delta{i} .* rmask{i};
        end
        
    end
end

%computer the gradient for the parameter W1,W2
for i=1:layernums,
    W_grad{i} = (delta{i+1} * a{i}')./n;
end

end
