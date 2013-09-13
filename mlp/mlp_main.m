% test the mlp.m
close all;
clear;
clc;


% load('dig.mat');
% X = X';
% 
% [D,N]=size(X)
% 
% perm = randperm(N);
% X = X(:,perm);
% y = y(perm);
% 
% Y=zeros(10,N);
% for i=1:N,
%     Y(y(i),i)=1;
% end
% 
% train_nums=round(0.7*N);
% X_train=X(:,1:train_nums);
% Y_train=Y(:,1:train_nums);
% 
% X_test=X(:,train_nums+1:N);
% Y_test=Y(:,train_nums+1:N);
% 
% save data X_train Y_train X_test Y_test;

load('data.mat');

[X, mu, sigma] = standardizing(X_train);
xnum = size(X,2);
trainnum = 1*xnum;
Y = Y_train;
X_train = X(:,1:trainnum);
Y_train = Y(:,1:trainnum);
X_val = X(:,trainnum+1:end);
Y_val = Y(:,trainnum+1:end);
[X_test,mu1, sigma1] = standardizing(X_test,'mu',mu,'sigma',sigma);


layers = 2;
hidnums = [1000];
outnums = 10;
lambda = 0;
% 
% active_fun=@sigmoid;
% active_fun_grad=@sigmoid_grad;
% grad = checkgradwithfinitediff(X_train,Y_train,layers,hidnums,outnums,active_fun,active_fun_grad)
rand('state',0)
model1 = mlp(X_train,Y_train,layers,hidnums,outnums,'lambda',lambda,'checkval',0,'dataval',X_val,'targetval',Y_val);

%[Y_new1, acc1] = predict(X_train,Y_train,model,@sigmoid);
[Y_new2, acc2] = predict(X_test,Y_test,model1,@sigmoid);
% 
% figure;
% plot(model.cost);

rand('state',0)
%test dropout
model2 = mlp(X_train,Y_train,layers,hidnums,outnums,'lambda',lambda,'checkval',0,...
            'dataval',X_val,'targetval',Y_val,'dropout',1);

%[Y_new1, acc1] = predict(X_train,Y_train,model,@sigmoid);
[Y_new2, acc2] = predict(X_test,Y_test,model2,@sigmoid);
% 
