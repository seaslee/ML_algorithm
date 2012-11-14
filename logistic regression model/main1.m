% the program gate
clear all;
close all;
clc;
%preprocess data set
load('breast-cancer_scale.mat');
n=size(Y,1);
for i=1:n,
    if Y(i)==2,
        Y(i)=0;
    else
        Y(i)=1;
    end
end
X=full(X);
X_train=X(1:400,:);
X_train=[ones(400,1),X_train];
Y_train=Y(1:400,:);
X_test=X(401:650,:);
X_test=[ones(250,1),X_test];
Y_test=Y(401:650,:);
%===============gradient descent to train parameter===================== 
theta=zeros(size(X_train,2),1);
eta=0.01;
iter_nums=10000;
iter_nums1=50;
epsilon=0.00001;
%[theta_opt,cost_fun_vals,is_con] = train_parameter(X_train,Y_train,theta,eta,iter_nums,epsilon);
%[Y_t,acc_t]=predict(X_train,Y_train,theta_opt);
%[Y_pre,acc]=predict(X_test,Y_test,theta_opt);
%acc_t
%theta_opt
%acc
%figure;
%plot(1:size(cost_fun_vals,1),cost_fun_vals,'markersize',5);
%xlabel('iteration number');
%ylabel('cost function value');
%title('the changeing cost function values');
%hold on;
%%==============stochastic gradient descent to train parameter============
%theta=zeros(size(X_train,2),1);
%eta=0.01;
%[theta_opt1,cost_fun_vals1,is_con] = train_parameter_sgd(X_train,Y_train,theta,eta,iter_nums1,epsilon);
%[Y_t1,acc_t1]=predict(X_train,Y_train,theta_opt1);
%[Y_pre1,acc1]=predict(X_test,Y_test,theta_opt1);
%acc_t1
%theta_opt1
%acc1
%is_con
%plot(1:size(cost_fun_vals1,1),cost_fun_vals1,'r','markersize',5);
%hold on;
%%=============gradient to train parameter with regularization============
%theta=zeros(size(X_train,2),1);
%eta=0.01;
%lambda=15;
%[theta_opt2,cost_fun_vals2,is_con] = train_parameter_reg(X_train,Y_train,theta,eta,iter_nums,epsilon,lambda);
%[Y_t2,acc_t2]=predict(X_train,Y_train,theta_opt2);
%[Y_pre2,acc2]=predict(X_test,Y_test,theta_opt2);
%acc_t2
%theta_opt2
%acc2
%plot(1:size(cost_fun_vals2,1),cost_fun_vals2,'k','markersize',5);
%legend('gd','sgd','gd_with_reg');
%print -dpng tr.png
%%==============learning rate effect ======================================
%theta=zeros(size(X_train,2),1);
%eta=0.01;
%iter_nums=1000;
%iter_nums1=50;
%epsilon=0.00001;
%[theta_opt,cost_fun_vals,is_con] = train_parameter(X_train,Y_train,theta,eta,iter_nums,epsilon);
%[Y_t,acc_t]=predict(X_train,Y_train,theta_opt);
%[Y_pre,acc]=predict(X_test,Y_test,theta_opt);
%figure;
%plot(1:size(cost_fun_vals,1),cost_fun_vals,'','markersize',5);
%xlabel('iteration number');
%ylabel('cost function value');
%title('the changeing cost function values');
%hold on;
%eta=1;
%[theta_opt,cost_fun_vals,is_con] = train_parameter(X_train,Y_train,theta,eta,iter_nums,epsilon);
%[Y_t,acc_t]=predict(X_train,Y_train,theta_opt);
%[Y_pre,acc]=predict(X_test,Y_test,theta_opt);
%plot(1:size(cost_fun_vals,1),cost_fun_vals,'r','markersize',5);
%xlabel('iteration number');
%ylabel('cost function value');
%title('the changeing cost function values');
%hold on;
%eta=50;
%[theta_opt,cost_fun_vals,is_con] = train_parameter(X_train,Y_train,theta,eta,iter_nums,epsilon);
%[Y_t,acc_t]=predict(X_train,Y_train,theta_opt);
%[Y_pre,acc]=predict(X_test,Y_test,theta_opt);
%plot(1:size(cost_fun_vals,1),cost_fun_vals,'k','markersize',5);
%xlabel('iteration number');
%ylabel('cost function value');
%title('the changeing cost function values');
%legend('rate=0.01','rate=1','rate=50',"boxon");
%print -dpng lr.png
%%==============sgd compare with shuffle and without shuffle============
theta=zeros(size(X_train,2),1);
eta=0.01;
iter_nums1=5;
[theta_opt1,cost_fun_vals1,is_con] = train_parameter_sgd(X_train,Y_train,theta,eta,iter_nums1,epsilon);
[Y_t1,acc_t1]=predict(X_train,Y_train,theta_opt1);
[Y_pre1,acc1]=predict(X_test,Y_test,theta_opt1);
acc_t1
theta_opt1
acc1
is_con
plot(1:size(cost_fun_vals1,1),cost_fun_vals1,'r','markersize',10);
hold on;
figure;
theta=zeros(size(X_train,2),1);
eta=0.01;
[theta_opt1,cost_fun_vals1,is_con] = train_parameter_sgd_without_shuffle(X_train,Y_train,theta,eta,iter_nums1,epsilon);
[Y_t1,acc_t1]=predict(X_train,Y_train,theta_opt1);
[Y_pre1,acc1]=predict(X_test,Y_test,theta_opt1);
acc_t1
theta_opt1
acc1
is_con
plot(1:size(cost_fun_vals1,1),cost_fun_vals1,'b','markersize',10);
hold on;
