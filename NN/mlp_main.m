% test the mlp.m
close all;
clear;
clc;
load('dig.mat');
[n,m]=size(X);
X=[ones(n,1),X];
Y=zeros(n,10);
for i=1:n,
    Y(i,y(i))=1;
end
train_nums=round(1*n);
X_train=X(1:train_nums,:);
Y_train=Y(1:train_nums,:);

hl_nums=30;
outl_nums=10;
eta=0.01;
iter_nums=5;
epsilon=0.0001;
active_fun=@sigmoid;
active_fun_grad=@sigmoid_grad;
m=size(X_train,2);
W1=zeros(hl_nums,m);
W2=zeros(outl_nums,hl_nums+1);
%[W1,W2]=init_para(W1,W2);
[W1_opt,W2_opt,cost_fun_vals,is_convg] = train_parameter(X_train,Y_train,W1,W2,eta,iter_nums,epsilon,active_fun,active_fun_grad);
Y_new = predict(X_train,W1_opt,W2_opt,active_fun);

pre_right=0;
pre_nums=size(Y_new,1);
for i=1:pre_nums,
    if isequal(Y_new(i,:),Y_train(i,:)),
        pre_right=pre_right+1;
    end
end
acc=pre_right/pre_nums;
fprintf('the accuracy is %f',acc);

figure;
plot(1:size(cost_fun_vals,1),cost_fun_vals,'markersize',10);

