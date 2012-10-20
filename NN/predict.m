%=============================================================================
%     FileName: mlp.m
%         Desc: a simple multiple layer perception implementation for binary classification
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-10-18 20:54:15
%      History:
%=============================================================================

function Y_new = predict(X_new,W1_opt,W2_opt,active_fun)
% Data: X_train, Y_train. add the (1,1,1,...,1)' to X's raw input
% parameter: w1 w2 is the parameter for first and second layer
% eta : the learning rate
Y_new = hypothesis(X_new,W1_opt,W2_opt,active_fun);
[row,column]=size(Y_new);
for i=1:row,
    [y,ind]=max(Y_new(i,:));
    for j=1:column,
        if j==ind,
            Y_new(i,j)=1;
        else
            Y_new(i,j)=0;
        end
    end
end

end


