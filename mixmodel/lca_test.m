%Test lca
close all;
clear;
clc;

dig = load('dig.mat');
X = dig.dig;
X = X';
K =10;

%lca test
threshold = 0.00001;
maxIterNum = 200;
initParamsMethod = 1;
[model, converged, iterNum, optObjVal, r] = lca(X, K, threshold, maxIterNum, initParamsMethod);
mu = model.mu;
figure(1);
for k=1:K,
    m = mu(:,k);
    dig = vec2mat(m,16);
    subplot(2,5,k);
    imshow(dig);
end

%kmeans test
%[ind, centroids, converged, iternum] = kmeans(X, K);
%mu = centroids;
%figure(2);
%for k=1:K,
    %m = mu(:,k);
    %dig = vec2mat(m,16);
    %subplot(2,5,k);
    %imshow(dig);
%end
