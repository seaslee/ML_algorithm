%=============================================================================
%     FileName: train_parameter.m
%         Desc: train linear regression models
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-10-14 22:38:37
%      History:
%=============================================================================
function [theta_opt,cost_fun_values,residual,is_con] = train_parameter(X_train,Y_train,theta,eta,iter_nums,epsilon)
n=size(Y_train,1);
% train the parameter theta using the gradient descent algorithm
cost_fun_values = zeros(iter_nums,1);
for i=1:iter_nums,
  theta = theta - eta*X_train'*(X_train*theta-Y_train)
  biasvec = X_train*theta-Y_train;
  cost_fun_values(i) = sqrt(biasvec'*biasvec/n);
  cost_fun_values(i)
  if i>=2 && abs(cost_fun_values(i-1)-cost_fun_values(i))<epsilon,
    is_con = 1;
    break
  end
end
if i==iter_nums,
    is_con=0;
end
theta_opt = theta;
residual=Y_train-X_train*theta_opt;
cost_fun_values=cost_fun_values(1:i);
end
