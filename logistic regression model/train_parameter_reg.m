%=============================================================================
%     FileName: train_parameter_reg.m
%         Desc: train parameter with regularization
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-10-17 09:23:11
%      History:
%=============================================================================
function [theta_opt,cost_fun_vals,is_con] = train_parameter_reg(X_train,Y_train,theta,eta,iter_nums,epsilon,lambda)
% the function use the gradient descent optimization method
% theta_opt : the theta value when gradient vanishes
% cost_fun_vals : the cost function value of each iteration
% is_con : 1 when algo convergence, 0 otherwise
m=size(X_train,1);
cost_fun_vals = zeros(iter_nums,1);
is_con = 0;
for i=1:iter_nums,
    Y_pr = sigmoid_fun(X_train*theta);
    theta = theta - 1/m*eta*(X_train'*(Y_pr-Y_train)+lambda*theta);
    theta(1) = theta(1) - 1/m*eta*(X_train'(1,:)*(Y_pr-Y_train));
    cost_fun_vals(i) = -1/m*(log(Y_pr)'*Y_train+log(1-Y_pr)'*(1-Y_train));
    %if i>=2 && abs(cost_fun_vals(i-1)-cost_fun_vals(i))<epsilon,
        %is_con = 1;
        %break
    %end
end
fprintf('the iteration number is %d\n',i);
theta_opt = theta;
cost_fun_vals=cost_fun_vals(1:i);
end

