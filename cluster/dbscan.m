function  [lables, iterNum] = dbscan(X, eps, minpts, varargin)
% agglomerative clustering
% Parameters:
%     -X: D*N dataset matrix whose column is sample and row is feature
%     -eps: Maximum radius of the neighbourhood
%     -minpts: Minimum number of points in an Eps-neighbourhood of that point
% Options:
%     -metric: used to computer the similarity matrix 
% Return:
%     -dendrogram: the tree of clusters
%     -iterNum: the iteration number of algorithm

s.metric = 'euclidean';
if ~isempty(varargin),
     s = handleoptions(s,varargin);
end

D = pdist(X', s.metric);
D = squareform(D);

N = size(X, 2); %numboer of samples
lables = -1 * ones(N,1); %-1 unvisited
C = 0;
iterNum = 1;

while 1,
    unvisited = find(lables == -1);
    if isempty(unvisited),
        break;
    else
        pidx = unvisited(1);
        neigx = find(D(pidx,:) < eps);
        n_neigx = length(neigx);
        if n_neigx < minpts,
            lables(pidx) = 0; %0 represents noisy point
        else
            C = C + 1;
            expandCluster();
        end
    end
    iterNum = iterNum + 1;  
    fprintf('Iteration number: %d\n', iterNum);
end

%========================================================
function  expandCluster()
lables(pidx) = C;

while ~isempty(neigx),
    nidx = neigx(1);
    neigx(1) = [];
    if lables(nidx) == -1,
        lables(nidx) = 0;
        neigx1 = find(D(nidx,:) < eps);
        n_neigx1 = length(neigx1);
        if n_neigx1 > minpts,
            neigx = [neigx neigx1];
        end
    end
    
    if lables(nidx)<1,
        lables(nidx) = C;
    end
end
end
%=========================================================       
end
