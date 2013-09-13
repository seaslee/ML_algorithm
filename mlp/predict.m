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

function [Y_new, acc] = predict(X_test,Y_test,model,active_fun)
% Data: X_train, Y_train. add the (1,1,1,...,1)' to X's raw input
% parameter: w1 w2 is the parameter for first and second layer
% eta : the learning rate
 
W = model.weights;
layernums = model.layers;
if model.dropout,
    for l=1:layernums,
        if l==1,
            W{l} = W{l} ;%* model.visualdroprate;
        else
            W{l} = W{l} * model.hiddendroprate;
        end
    end
end

z{1} = X_test;
a{1} = X_test;
for i=1:layernums,
    a{i} = [ones(1,size(a{i},2)); a{i}];
    z{i+1} = W{i} * a{i};
    a{i+1} = active_fun(z{i+1});
end
Y_new = a{i+1};
[D,N]=size(Y_new);
Y_pred = zeros(D, N);
for i=1:N,
    [y,ind]=max(Y_new(:,i));
    Y_pred(ind,i) = 1;
    for j=1:D,
        if j==ind,
            Y_new(j,i)=1;
        else
            Y_new(j,i)=0;
        end
    end
end

pre_right=0;
pre_nums=size(Y_test,2);
for i=1:pre_nums,
    if isequal(Y_new(:,i),Y_test(:,i)),
        pre_right=pre_right+1;
    end
end
acc=pre_right/pre_nums;
fprintf('the accuracy is %f\n',acc);

end


