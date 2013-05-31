close all;
clear;
clc;

%randomly sample from the Gaussian Mixture distribution
MU = [1 2;-2 -3;-3 4];
SIGMA = cat(3,[2 0;0 .5],[1 0;0 1],[1 0;0 1]);
p = ones(1,3)/3;
obj = gmdistribution(MU,SIGMA,p);
%ezcontour(@(x,y)pdf(obj,[x y]),[-10 10],[-10 10])
hold on;
Y = random(obj,2000);
%scatter(Y(:,1),Y(:,2),25,'.')

 
X = Y';
K = 3;

%kmeans test
figure(1);
[ind, centroids, converged, iternum] = kmeans(X, K);
for k=1:K,
    index = find(ind==k);
    X_k = X(:,index);
    if k==1,
        scatter(X_k(1,:),X_k(2,:),20,'o','r');
    elseif k==2,
        scatter(X_k(1,:),X_k(2,:),20,'+','g');
    else
        scatter(X_k(1,:),X_k(2,:),20,'*','k');
    end
    hold on;
end

%kmedois test
figure(2);
[ind, centroids, converged, iternum] = kmedois(X, K);
for k=1:K,
    index = find(ind==k);
    X_k = X(:,index);
    if k==1,
        scatter(X_k(1,:),X_k(2,:),20,'o','r');
    elseif k==2,
        scatter(X_k(1,:),X_k(2,:),20,'+','g');
    else
        scatter(X_k(1,:),X_k(2,:),20,'*','k');
    end
    hold on;
end

% %spectrum clustering test
figure(3);
%computer similarity matrix
N = size(X,2);
W = zeros(N);
for i=1:N,
    for j=1:N,
        if j~=i,
            x1 = X(:,i);
            x2 = X(:,j);
            %W(i,j) = x1'*x2./(norm(x1)*norm(x2));
            W(i, j) = exp(-norm(x1-x2)^2/2);
        else
            W(i,j)=0;
        end
    end
end

[ind, converged, iterNum] = spectrumcluster(W, K, 'normalized',2);
for k=1:K,
    index = find(ind==k);
    X_k = X(:,index);
    if k==1,
        scatter(X_k(1,:),X_k(2,:),20,'o','r');
    elseif k==2,
        scatter(X_k(1,:),X_k(2,:),20,'+','g');
    else
        scatter(X_k(1,:),X_k(2,:),20,'*','k');
    end
    hold on;
end

% test agglomerative clustering
load fisheriris
%X = meas(1:5,:);
[denMat, iterNum] = agglcluster(meas');
figure(4);
dendrogram(denMat);

% test dbscan 
eps = 0.5;
minpts = 30;
[ind, iterNum] = dbscan(X, eps, minpts);
figure(5);
for k=1:K,
    index = find(ind==k);
    X_k = X(:,index);
    if k==1,
        scatter(X_k(1,:),X_k(2,:),20,'o','r');
    elseif k==2,
        scatter(X_k(1,:),X_k(2,:),20,'+','g');
    else
        scatter(X_k(1,:),X_k(2,:),20,'*','k');
    end
    hold on;
end