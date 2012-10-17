%=============================================================================
%     FileName: sigmoid_fun.m
%         Desc: sigmoid funcion which is 1/(1+e(-z))
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-10-17 09:06:17
%      History:
%=============================================================================
function Y = sigmoid_fun(Z)
Y = 1./(1+exp(-Z));
end
