%=============================================================================
%     FileName: pca.m
%         Desc: perform the computation of PCA
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-11-14 16:26:33
%      History:
%=============================================================================

function [U,S]=pca(X)
% The function performs the computation of principle component analysis(pca).
% args: 
%     X is a matrix of data set where each row is a feature variable
%     and each column a data instance.
% return:
%     U is a matrix whose columns are eigenvectors.
%     S is a diagnoal matrix of eigenvalues.
[d,n]=size(X);
mu=mean(X);
X=bsxfun(@minus,X,mu);

if d<n,
    %covariance matrix
    Sigma=(1/n)*(X*X');
    [U,S,V]=svd(Sigma);
else
    %covariance matrix
    Sigma=(1/n)*(X'*X);
    [U,S,V]=svd(Sigma);
    U=X*U;
    n=size(U,2);
    %normalize to the unit vector for each column in U
    for i=1:n,
        U(:,i)=U(:,i)/norm(U(:,i));
    end
end
    
