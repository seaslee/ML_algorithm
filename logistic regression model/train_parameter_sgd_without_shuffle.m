%=============================================================================
%     FileName: train_parameter_without_shuffle.m
%         Desc: sgd without shuffle
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-10-17 22:42:29
%      History:
%=============================================================================

function [theta_opt,cost_fun_vals,is_con] = train_parameter_sgd_without_shuffle(X_train,Y_train,theta,eta,iter_nums,epsilon)
% the function use the stochatic gradient descent optimization method
cost_fun_vals = [];
m = size(X_train,1)
X_index= [1:m].';
Y_pr=zeros(m,1);
is_con=0;
iter=1;
for i=1:iter_nums,
    %% use Fisher_Yates algo to shuffle the index array of X_train 
    %for k=m:-1:1,
        %l=round(rand(1)*k)+1;
        %if l>m,
            %l=l-1;
        %end
        %tmp=X_index(k);
        %X_index(k)=X_index(l);
        %X_index(l)=tmp;
    %end
    for j=1:m,
        eta_j=eta/(1+eta*j);
        Y_pr(j) = sigmoid_fun(X_train(j,:)*theta);
        theta = theta - eta_j*X_train(j,:)'*(Y_pr(j)-Y_train(j));
        cost = -1/m*(log(Y_pr)'*Y_train+log(1-Y_pr)'*(1-Y_train));
        cost_fun_vals=[cost_fun_vals;cost];
        %if iter>=2 && abs(cost_fun_vals(iter-1)-cost_fun_vals(iter))<epsilon,
            %is_con = 1;
            %break
        %end
        %iter=iter+1;
    end
end


fprintf('the iteration number is %d\n',i);
theta_opt = theta;

end
