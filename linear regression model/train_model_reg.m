%=============================================================================
%     FileName: train_model_reg.m
%         Desc: train model with regularization
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-10-16 09:05:52
%      History:
%=============================================================================

function [best_theta,best_k,best_lambda,cost_fun_values,residual,is_con,err_k]=train_model_reg(X,Y,theta,eta,iter_nums,epsilon,fold)
n=size(X,1); %the total numbers in training set
foldnum=n/fold; %number in each fold
err_f=zeros(fold,1);
err_values=zeros(10,1);
init_lambda=1;
for k=1:10,
    k
    %n-flod validation
    for lambda=init_lambda:5:200,
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
            [theta_opt,cost_fun_values,residual,is_con]=train_parameter_reg(X_train,Y_train,theta,lambda,eta,iter_nums,epsilon);
            [Y_pre,err]=predict(X_test,Y_test,theta_opt,k,X_train,residual);
            err_f(i)=err;
        end
        err_avg1=mean(err_f);
        if lambda==init_lambda,
            err_l=err_avg1;
        else
            err_l=[err_l,err_avg1];
        end
    end
    if k==1,
        err_k=err_l;
    else
        err_k=[err_k;err_l];
    end
end
% use all training data to learn parameter
err_k
minv=min(min(err_k));
[best_k,best_lambda]=find(err_k==minv);
best_lambda=5*best_lambda;
[best_theta,cost_fun_values,residual,is_con]=train_parameter_reg(X,Y,theta,best_lambda,eta,iter_nums,epsilon);

end
