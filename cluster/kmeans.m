function [idx, centroids, converged, iterNum] = kmeans(X, K, maxiterNum, threshold)
% kmeans method for clustering
% Parameters:
%     -X: dataset matrix whose column is sample and row is feature
%     -K: number of clusters
% Options:
%     -maxiterNum: maximum number of iteration
%     -threshold: convergence threshold
% Return:
%     -idx: cluster's index for every sample
%     -centroids: centroids of K clusters
%     -converged: true if algorithm converged, otherwise, false
%     -iterNum: number of iteration of algorithm
% Note that the number of parameters can 2 or 4. CAN'T be other number.

if nargin==2,
    maxiterNum=200;
    threshold = 1.0e-9;
elseif nargin==4,
    ;
else
    error('invalaid parameters');
end
[D, N] = size(X)
centroids = initCentroids();
iterNum = 0;
converged = 0;
idx = zeros(N, 1);
d = zeros(N, K);

while iterNum < maxiterNum,
    iterNum = iterNum + 1;
    %kmeans iteration
    %assign every sample to cluster according to current centroids
    for n = 1:N,
        for k = 1:K,
            d(n,k) = norm(X(:,n) - centroids(:,k));
        end
    end
    [~, idx] = min(d, [], 2);
 
    %computer the centroids with the new assignment of samples
    for k = 1:K,
        centroids(:,k) = mean(X(:,idx==k), 2);
    end
    
    curObj = computerObjVal();
    fprintf('Iterative number: %d, Objective Value: %f\n',iterNum, curObj);
    if iterNum>1 && abs(curObj-preObj) < threshold,
        converged = 1;
        break;
    end
    
    preObj = curObj;
end

function centroids = initCentroids()
%randomly select K samples from X as centroids 
N
perm = randperm(N);
centroids = X(:,perm(1:K));
end

function objVal = computerObjVal()
%computer objective function value
d1 = zeros(N, K);
for nn = 1:N,
    for kk = 1:K,
        d1(nn,kk) = norm(X(:,nn) - centroids(:,kk));
    end
end
objVal = sum(sum(d1));
end 

end
        

        
   