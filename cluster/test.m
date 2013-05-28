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


%gmm test
% threshold =.00001;
% maxIterNum = 200;
% initParamsMethod = 0;
% [model, converged, iterNum, optObjVal, r] = gmm(X, K, threshold, maxIterNum, initParamsMethod);
% 
% [~, ind] = max(r, [], 2);
% figure(1);
% for k=1:K,
%     index = find(ind==k);
%     X_k = X(:,index);
%     if k==1,
%         scatter(X_k(1,:),X_k(2,:),20,'o','r');
%     elseif k==2,
%         scatter(X_k(1,:),X_k(2,:),20,'+','g');
%     else
%         scatter(X_k(1,:),X_k(2,:),20,'*','k');
%     end
%     hold on;
% end
% 
% %kmeans test
figure(2);
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
figure(3);
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
