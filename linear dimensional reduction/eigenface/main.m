close all;
clear;
clc;

[X,Y]=getImage('att_faces');
X=standardizing(X);
[U,S]=pca(X);
m=300;
U=U(:,1:300);

%projected data
X=U'*X;
%shuffer the data set
n=size(X,2);
pind=randperm(n);
X=X(:,pind);
Y=Y(pind);
%face recognition using knn

%training data
X_train=X(:,1:200);
X_test=X(:,201:end);
Y_train=Y(1:200);
Y_test=Y(201:end);

K_min=1;
K_max=5;
fold=4;
k=train_model(X_train,Y_train,fold,K_min,K_max)
test_nums=size(X_test,2);
Y_pr=[];
for i=1:test_nums,
    x=X_test(:,i);
    kindex=knn(X_train,x,k);
    labels=Y_train(kindex);
    label=mode(labels);
    Y_pr=[Y_pr;label];
end

acc=sum(Y_test==Y_pr)/size(Y_test,1);
fprintf('the acc is %f\n',acc);

%plot confusion matrix
[confusion_matrix]=compute_confusion_matrix(Y_test,Y_pr,40);
    
