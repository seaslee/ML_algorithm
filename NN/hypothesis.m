%=============================================================================
%     FileName: hypothesis.m
%         Desc: hypothesis function
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-10-18 20:41:39
%      History:
%=============================================================================
function Y = hypothesis(X,W1,W2,active_fun)
%hypothesis function
n=size(X,1);
hl_vals=active_fun(W1*X');
hl_vals=hl_vals';
hl_vals=[ones(n,1),hl_vals];
Y = active_fun(W2*hl_vals');
Y=Y';
end
