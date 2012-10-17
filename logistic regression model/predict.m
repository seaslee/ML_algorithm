%=============================================================================
%     FileName: predict.m
%         Desc: predictive function for logistic regression
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-10-17 09:00:30
%      History:
%=============================================================================
function [Y_pre,acc]=predict(X,Y,theta)
% y=sigmoid(X*theta)
% X : the data set, row is #feature, column is #data
% Y : the label of the data set
% theta : the parameter
Y_pre = sigmoid_fun(X*theta);
n=size(Y_pre,1);
for i=1:n,
    if Y_pre(i)>=0.5,
        Y_pre(i)=1;
    else
        Y_pre(i)=0;
    end
end
acc=sum(Y_pre==Y)/size(Y,1);

end
