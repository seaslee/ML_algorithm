%=============================================================================
%     FileName: init_para.m
%         Desc: initialize parameter for bp-network
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-10-18 20:36:54
%      History:
%=============================================================================
function [W1,W2] = init_para(W1,W2)
% the method to initialize parameters is from ML class of A.Ng
[r1,c1]=size(W1);
[r2,c2]=size(W2);
epsilon_init1=sqrt(6)/sqrt(r1+c1);
epsilon_init2=sqrt(6)/sqrt(r2+c2);
W1=rand(r1,c1)*2*epsilon_init1-epsilon_init1;
W2=rand(r2,c2)*2*epsilon_init2-epsilon_init2;
end
