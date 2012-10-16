%=============================================================================
%     FileName: scale.m
%         Desc: scale data 
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-10-15 20:12:20
%      History:
%=============================================================================

function X_scale=scale(X)

n=size(X,2);
mu=mean(X);
sigma=std(X);
for i=1:n,
    X_scale(:,i)=(X(:,i)-mu(i))/sigma(i);
end

end

