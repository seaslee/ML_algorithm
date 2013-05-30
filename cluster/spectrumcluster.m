function [idx, converged, iterNum] = spectrumcluster(W, K, varargin)
% simple implementation for spectrumClustering
% spectrum clustering algorithm in ¡°A Tutorial on Spectrum clustring¡±(Luxburg)
% Parameters:
%     -W: Similarity matrix of the graph
%     -K: number of clusters
% Options:
%     -maxiterNum: maximum number of iteration
%     -threshold: convergence threshold
%     -normalized: 0 unnormalized, 1 normalized, 2 normalized
% Return:
%     -idx: cluster's index for every sample
%     -converged: true if algorithm converged, otherwise, false
%     -iterNum: number of iteration of algorithm

s.maxiterNum = 200;
s.threshold = 1.0e-8;
s.normalized = 0;

if ~isempty(varargin),
    s = handleoptions(s,varargin);
end

N = size(W,2);
d = sum(W);
d(d==0) = 1; %avoid zeros-divied
D = diag(d);
% Laplacian matrix
L = D - W;

if s.normalized == 0,
    [V, ~] = eigs(L, K, 'sm');
elseif s.normalized == 1,
    sd = diag(d.^(-1));
    L = eye(N) - sd*W;
    [V, ~] = eigs(L, K, 'sm');
    %normalization rows
elseif s.normalized == 2,
    %ref: "On Spectrum clustering: Analysis and an algorith"(Ang 2002)
    sd = diag(d.^(-1/2));
    L =  eye(N) - sd*W*sd;
    [V, ~] = eigs(L, K, 'sm');
    %normalization rows
    squareSumOfCols = dot(V,V);
    V = bsxfun(@rdivide, V, squareSumOfCols);
else
    error('wrong normalized parameter value',s.normalized);
end

%kmeans cluster for reduced data
[idx, centroids, converged, iterNum] = kmeans(V', K, s.maxiterNum, s.threshold);

end



