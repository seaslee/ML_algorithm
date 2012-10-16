%=============================================================================
%     FileName: knn.m
%         Desc: computer knn for the test
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-10-15 19:25:12
%      History:
%=============================================================================

function index=knn(X,x,K)
n=size(X,1); 
dist=zeros(n,1);
for i=1:n,
    dist(i)=norm(X(i,:)-x);
end
[x,ix]=sort(dist);
index=ix(1:K);
end

