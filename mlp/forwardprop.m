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
function [z, a, cost, rmask] = forwardprop(x,y,layernums,W,active_fun,varargin)
%forward-propagation to computer value of every node in the net
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

z{1} = x;
a{1} = x;
rmask = {};
%dropout
% if dropout,
%     rmask{1} = rand(size(a{1})) > visualdroprate;
%     a{1} = a{1} .* rmask{1};
% end

for i=1:layernums,
    a{i} = [ones(1,size(a{i},2)); a{i}];
    z{i+1} = W{i} * a{i};
    a{i+1} = active_fun(z{i+1});
    if dropout && i~=layernums,
        rmask{i+1} = (rand(size(a{i+1})) > hiddendroprate);
        a{i+1} = a{i+1} .* rmask{i+1};  
    end
end

f = a{i+1};
y_train = f;

assert(sum(sum(y_train<0 &y_train>1))==0,'wrong output');
cost = y.*log(y_train+eps)+(1-y).*log(1-y_train+eps);
cost = - sum(sum(cost))./n;
end
