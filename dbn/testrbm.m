%test rbm
close all;
clear;
clc;

% ===============================================
% load minist dataset modifying from Hinton's code
% http://www.cs.toronto.edu/~hinton/code/makebatches.m
digitdata=[]; 
targets=[]; 
load '../minist/digit0'; digitdata = [digitdata; D]; targets = [targets; repmat([1 0 0 0 0 0 0 0 0 0], size(D,1), 1)];  
load '../minist/digit1'; digitdata = [digitdata; D]; targets = [targets; repmat([0 1 0 0 0 0 0 0 0 0], size(D,1), 1)];
load '../minist/digit2'; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 1 0 0 0 0 0 0 0], size(D,1), 1)]; 
load '../minist/digit3'; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 1 0 0 0 0 0 0], size(D,1), 1)];
load '../minist/digit4'; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 1 0 0 0 0 0], size(D,1), 1)]; 
load '../minist/digit5'; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 1 0 0 0 0], size(D,1), 1)];
load '../minist/digit6'; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 1 0 0 0], size(D,1), 1)];
load '../minist/digit7'; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 1 0 0], size(D,1), 1)];
load '../minist/digit8'; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 0 1 0], size(D,1), 1)];
load '../minist/digit9'; digitdata = [digitdata; D]; targets = [targets; repmat([0 0 0 0 0 0 0 0 0 1], size(D,1), 1)];
digitdata = digitdata/255;

traindata = digitdata;
traintarget = targets;

totnum=size(digitdata,1);
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=totnum/100;
numdims  =  size(digitdata,2);
batchsize = 100;
batchdata = zeros(batchsize, numdims, numbatches);
batchtargets = zeros(batchsize, 10, numbatches);

for b=1:numbatches
  batchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  batchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear digitdata targets;

digitdata=[];
targets=[];
load '../minist/test0'; digitdata = [digitdata; D(1:100,:)]; targets = [targets; repmat([1 0 0 0 0 0 0 0 0 0], size(D,1), 1)]; 
load '../minist/test1'; digitdata = [digitdata; D(1:100,:)]; targets = [targets; repmat([0 1 0 0 0 0 0 0 0 0], size(D,1), 1)]; 
load '../minist/test2'; digitdata = [digitdata; D(1:100,:)]; targets = [targets; repmat([0 0 1 0 0 0 0 0 0 0], size(D,1), 1)];
load '../minist/test3'; digitdata = [digitdata; D(1:100,:)]; targets = [targets; repmat([0 0 0 1 0 0 0 0 0 0], size(D,1), 1)];
load '../minist/test4'; digitdata = [digitdata; D(1:100,:)]; targets = [targets; repmat([0 0 0 0 1 0 0 0 0 0], size(D,1), 1)];
load '../minist/test5'; digitdata = [digitdata; D(1:100,:)]; targets = [targets; repmat([0 0 0 0 0 1 0 0 0 0], size(D,1), 1)];
load '../minist/test6'; digitdata = [digitdata; D(1:100,:)]; targets = [targets; repmat([0 0 0 0 0 0 1 0 0 0], size(D,1), 1)];
load '../minist/test7'; digitdata = [digitdata; D(1:100,:)]; targets = [targets; repmat([0 0 0 0 0 0 0 1 0 0], size(D,1), 1)];
load '../minist/test8'; digitdata = [digitdata; D(1:100,:)]; targets = [targets; repmat([0 0 0 0 0 0 0 0 1 0], size(D,1), 1)];
load '../minist/test9'; digitdata = [digitdata; D(1:100,:)]; targets = [targets; repmat([0 0 0 0 0 0 0 0 0 1], size(D,1), 1)];
digitdata = digitdata/255;

testdata = digitdata;
testtarget = targets;

totnum=size(digitdata,1);
fprintf(1, 'Size of the test dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=totnum/100;
numdims  =  size(digitdata,2);
batchsize = 100;
testbatchdata = zeros(batchsize, numdims, numbatches);
testbatchtargets = zeros(batchsize, 10, numbatches);

for b=1:numbatches
  testbatchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  testbatchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear digitdata targets;


%%% Reset random seeds 
rand('state',sum(100*clock)); 
randn('state',sum(100*clock)); 

%========train RBM model==============================
X = traindata';
visnum = size(X, 1);
hidnum = 100;
maxiternum = 5000;
rate = 0.1;
epsilon = 1e-6;
tic;
% [model, iternum] = rbm(X,hidnum);  
% toc;
% % ========visual==============================
% rp=randperm(1000);
% test = testdata(rp(1:100),:);
% figure;
% displayData(test,28);
% Y = RBMGenData(model,test',1000);
% figure ;
% displayData(Y',28);
% 
% figure;
% wei = model.weight';
% displayData(wei,28);

layernums = 2;
hidnums = [200,200];
model = dbn(X,layernums,hidnums);