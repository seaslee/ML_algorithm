%=============================================================================
%     FileName: standardizing.m
%         Desc: standardizing the columns in matrix X
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-11-14 16:46:57
%      History:
%=============================================================================

function X=standardizing(X)
% X=standardizing(X) performs the standardizing computation,which 
% makes the colums in return matrix X have the zero expectation  
% and unit variance.

mu=mean(X);
sigma=std(X);
X=bsxfun(@rdivide,bsxfun(@minus,X,mu),sigma);
end
