%randomly sample from the Gaussian Mixture distribution
MU = [1 2;-2 -3;-3 4];
SIGMA = cat(3,[2 0;0 .5],[1 0;0 1],[1 0;0 1]);
p = ones(1,3)/3;
obj = gmdistribution(MU,SIGMA,p);
%ezcontour(@(x,y)pdf(obj,[x y]),[-10 10],[-10 10])
hold on;
Y = random(obj,2000);
%scatter(Y(:,1),Y(:,2),25,'.')

%gmm 
X = Y';
K = 3;
threshold =.00001;
maxIterNum = 200;
initParamsMethod = 1;
[model, converged, iterNum, optObjVal, r] = gmm(X, K, threshold, maxIterNum, initParamsMethod);

[~, ind] = max(r, [], 2);
%ind = kmeans(X',K);
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

