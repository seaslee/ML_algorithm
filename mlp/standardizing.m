%=============================================================================
%     FileName: standardizing.m
%         Desc: standardizing the columns in matrix X

%      Version: 0.0.1
%   LastChange: 2012-11-14 16:46:57
%      History:
%=============================================================================

function [X, mu, sigma]=standardizing(X,varargin)
% X=standardizing(X) performs the standardizing computation,which 
% makes the colums in return matrix X have the zero expectation  
% and unit variance.
% ============================
% Author: XuXinchao
% Email: xxinchao@gmail.com
% HomePage: http://webdancer.is-programmer.com
s.mu=mean(X,2);
s.sigma=max(std(X,0,2), eps);
if ~isempty(varargin),
    s = handleoptions(s,varargin);
end

mu = s.mu;
sigma = s.sigma;
X=bsxfun(@rdivide,bsxfun(@minus,X,mu),sigma);
end
