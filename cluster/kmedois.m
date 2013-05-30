function [idx, centroids, converged, iterNum] = kmedois(X, K, maxiterNum, threshold)
% kmedois method for clustering with PAM algorithm(Partitioning Around Medoids)
% Parameters:
%     -X: d*n  dataset matrix whose column is sample and row is feature
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
centindx = initCentroids();
lastcentindx = centindx;
iterNum = 0;
converged = 0;
idx = zeros(N, 1);
d = zeros(N, K);

%pre-computer the distances between every sample
squareSumOfCols = dot(X, X);
disMat = bsxfun(@plus, squareSumOfCols, squareSumOfCols') - 2*(X'*X);

while iterNum < maxiterNum,
    iterNum = iterNum + 1;
    %kmeans iteration
    %assign every sample to cluster according to current centroids
    [~, idx] = min(disMat(centindx,:));
 
    %computer the centroids with the new assignment of samples
    isswap = 1;
    r =  sparse(1:N, idx, 1, N, K, N); % r_{n,k} represents x_n in k_th
    sumDis2Cent = disMat*r;
    [~, centindx] = min(sumDis2Cent);
    % check whether to swap mediois and other point
    if any(centindx ~= lastcentindx),
        isswap = 1;
    else
        isswap = 0;
        lastcentindx = centindx;
    end

    curObj = computerObjVal();
    fprintf('Iterative number: %d, Objective Value: %f\n',iterNum, curObj);
    if isswap == 0 || (iterNum>1 && abs(curObj-preObj) < threshold),
        centroids = X(:,centindx);
        converged = 1;
        break;
    end
    
    preObj = curObj;
end

function centindx = initCentroids()
%randomly select K samples from X as centroids 
centindx = randsample(N, K)';
end

function objVal = computerObjVal()
%computer objective function value
objVal = sum(sum(bsxfun(@times, r, disMat(:,centindx))));
objVal = full(objVal);
end 

end
        
   