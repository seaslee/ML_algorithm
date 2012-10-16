%=============================================================================
%     FileName: predict.m
%         Desc: predictive function for the linear regression model
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-10-15 18:47:21
%      History:
%=============================================================================

function [Y_pre,err]=predict(X_test,Y_test,theta,K,X_train,residual)
%computer the k-nearest-neigbor for X_test
n=size(X_test,1);
for i=1:n,
    x=X_test(i,:);
    knnidx=knn(X_train,x,K);
    Y_pre(i)=x*theta+sum(residual(knnidx))/K;
end

err=sqrt(norm(Y_test-Y_pre')^2/n);

end

