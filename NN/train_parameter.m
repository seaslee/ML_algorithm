%=============================================================================
%     FileName: train_parameter.m
%         Desc: train parameter for bp-network
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-10-18 20:14:44
%      History:
%=============================================================================
function [W1_opt,W2_opt,cost_fun_vals,is_convg] = train_parameter(X_train,Y_train,W1,W2,eta,iter_nums,epsilon,active_fun,active_fun_grad)
% use stochastic gradient descent to train parameter
N=size(X_train,1);
cost_fun_vals=[];
is_convg=0;
for i=1:iter_nums,
    perm=randperm(N);
    X_train=X_train(perm,:);
    Y_train=Y_train(perm,:);
    for n=1:N,
        (i-1)*N+n
        [W1_grad,W2_grad] = mlp_backprop(X_train(n,:),Y_train(n,:),W1,W2,active_fun,active_fun_grad);
        W1=W1-eta*W1_grad;
        W2=W2-eta*W2_grad
	cost_fun_val=computer_cost_fun(X_train,Y_train,W1,W2,active_fun);
	cost_fun_vals=[cost_fun_vals;cost_fun_val];
    end
    %
    %if i>2 && abs(cost_fun_vals(i-1)-cost_fun_vals(i))<epsilon,
        %is_convg=1;
        %break;
    %end
end
W1_opt=W1;
W2_opt=W2;
end

function [W1_grad,W2_grad] = mlp_backprop(x,y,W1,W2,active_fun,active_fun_grad)
% backpropagation algo to computer gradient

%forward-propagation to computer value of every node in the net

z2=W1*x';
a2=active_fun(z2);
a2=[1;a2];
z3=W2*a2;
a3=active_fun(z3);

%computer the delta value for every node in the net
a3=a3';
delta3=a3-y;
tmp=W2'*delta3';
tmp=tmp(2:end);
delta2=tmp.*active_fun_grad(z2);
%computer the gradient for the parameter W1,W2
W1_grad=delta2*x;
W2_grad=delta3'*a2';
end

function cost_val=computer_cost_fun(X_train,Y_train,W1,W2,active_fun)
%computer the cost function value
N=size(X_train,1);
Y=hypothesis(X_train,W1,W2,active_fun);
val=Y_train.*log(Y)+(1-Y_train).*log(1-Y);
cost_val=-sum(sum(val))/N;
end


