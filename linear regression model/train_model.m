%=============================================================================
%     FileName: train_model.m
%         Desc: train model for linear regression with k-neigbor
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-10-14 23:04:59
%      History:
%=============================================================================

function [best_theta,best_k,cost_fun_values,residual,is_con,err_values]=train_model(X,Y,theta,eta,iter_nums,epsilon,fold)
n=size(X,1); %the total numbers in training set
foldnum=n/fold; %number in each fold
err_k=0; 
err_f=zeros(fold,1);
err_values=zeros(10,1);
for k=1:10,
    %n-flod validation
    for i=1:fold,
        % put the data in part for n-fold validation
        test_index=1+(i-1)*foldnum:i*foldnum;
        if i==1,
            train_index=i*foldnum+1:n;
        elseif i==5,
            train_index=1:(i-1)*foldnum;
        else
            train_index=[1+(i-2)*foldnum:(i-1)*foldnum,1+i*foldnum:n];
        end
        X_train=X(train_index,:);
        Y_train=Y(train_index);

        X_test=X(test_index,:);
        Y_test=Y(test_index);
        size(X_train)
        [theta_opt,cost_fun_values,residual,is_con]=train_parameter(X_train,Y_train,theta,eta,iter_nums,epsilon);
        [Y_pre,err]=predict(X_test,Y_test,theta_opt,k,X_train,residual);
        err_f(i)=err;
    end
    err_avg=mean(err_f);
    err_values(k)=err_avg;
    if k==1,
        err_k=err_avg;
        best_k=k;
    elseif err_avg<err_k,
        err_k=err_avg;
        best_k=k;
    end
end
% use all training data to learn parameter
[best_theta,cost_fun_values,residual,is_con]=train_parameter(X,Y,theta,eta,iter_nums,epsilon);

end
