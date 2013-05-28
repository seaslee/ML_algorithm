function [idx, centroids, converged, iterNum] = kmedois(X, K, maxiterNum, threshold)
% kmedois method for clustering with PAM algorithm(Partitioning Around Medoids)
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
% =====================================================================
% The distance function in kmedois can be any one of dissimiliarity functions, 
% Not only limit to Euclidean distance function, but Euclidean distance
% function is used here.

if nargin==2,
    maxiterNum=200;
    threshold = 1.0e-9;
elseif nargin==4,
    ;
else
    error('invalaid parameters');
end
[~, N] = size(X);
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
    isswap = 1;
    for k = 1:K,
        X_k = X(:,idx==k);
        if isempty(X_k),
            id = randperm(N);
            centroids(:,k) = X(:, id(1));
        else
            n_k = size(X_k, 2);
            dk = zeros(n_k);
            for i=1:n_k,
                for j=1:n_k,
                        dk(i,j) = norm(X_k(:,i)-X_k(:,j));
                end
            end
            dd = sum(dk);
            [~, id1] = min(dd);
            % check whether to swap mediois and other point
            if sum(centroids(:,k) == X_k(:,id1)) == n_k,
                isswap = 0;
            else
                centroids(:, k) = X_k(:, id1);
                isswap = 1;
            end
    end
    end
    
    curObj = computerObjVal();
    fprintf('Iterative number: %d, Objective Value: %f\n',iterNum, curObj);
    if isswap == 0 || (iterNum>1 && abs(curObj-preObj) < threshold),
        converged = 1;
        break;
    end
    
    preObj = curObj;
end

function centroids = initCentroids()
%randomly select K samples from X as centroids 
perm = randperm(N);
idxx = perm(1:K);
centroids = X(:,idxx);
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
        
   