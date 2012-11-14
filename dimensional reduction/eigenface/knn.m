%=============================================================================
%     FileName: knn.m
%         Desc: computer k-nearest neighbor in the matrix for vector x
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-10-15 19:25:12
%      History:
%=============================================================================

function indices=knn(X,x,K)
%perform computation of K-nearest neighbor for x in matrix X
%args:
%   X is a matrix where each column is a training instance.
%   x is a vector which is a training instance
%   K is parameter to control how many nearest neighbors to find for x in X
%return:
    %index is the indices of K-nearest neighbor of x in X
n=size(X,2); 
dist=zeros(n,1);
for i=1:n
    dist(i)=norm(X(:,i)-x);
end
[x,ix]=sort(dist);
indices=ix(1:K);
end

