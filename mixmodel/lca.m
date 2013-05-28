function [model, converged, iterNum, optObjVal, r] = lca(X, K, threshold, maxIterNum, initParamsMethod)
% MLE for LCA(Latent Class Analysis) with EM algorithm
%   parameters:
%   -X: the dataset matrix whose column is sample and row is feature
%   -K: number of components of GMM
%   -threshold: convergence threshold
%   -maxIterNum: maximum iterative number
%   -initParams: 0 randomly initialize, 1 kmeans method to initialize 
%   Return:
%   -model: the gmm model to fit the data X
%   -converged: true if EM algorithm is converged, otherwise, false
%   -iterNum: iterative number of EM algorithm
%   -optObjVal: final objective function value

[pi, mu] = initParams(X, K, initParamsMethod);

[D, N] = size(X);
converged = 0;
iterNum = 0;
r = zeros(N,K);
preObjVal = computerObjVal(X, K, pi, mu);

%EM iteration
while iterNum < maxIterNum,
    iterNum = iterNum + 1;
    %E step: computer responsibility of k component for nth sample
    for n=1:N,
        for k=1:K,
            r(n,k) = pi(k) * computerBernProb(X(:,n), mu(:,k));
        end
    end
    z = sum(r,2);
    for n=1:N,
        for k=1:K,
            r(n,k) = r(n,k)./ z(n);
        end
    end
    
    %M step: maximum the log likelihood 
    Nk = sum(r);
    %mu = bsxfun(@rdivide, X*r, repmat(Nk, D, 1));
    mu = X*r*diag(1./Nk);
    pi = Nk / N; 
    curObjVal = computerObjVal(X, K, pi, mu);
    
    fprintf('Iterative number: %d, Objective Value: %f\n',iterNum, curObjVal);
    
    if abs(curObjVal-preObjVal) < threshold,
        converged = 1;
        break;
    end
    
    preObjVal = curObjVal;
end

model.K = K;
model.weight = pi;
model.mu = mu;
optObjVal = curObjVal;

end


function [pi, mu] = initParams(X, K, initParamsMethod)
% initialize paramters 
[D,N] = size(X);
if initParamsMethod ==0,
    %randomly initialize parameter
    perm = randperm(N);
    mu = X(:,perm(1:K));
    pi = zeros(K, 1);

    d = zeros(N,K);
    for n=1:N,
        for k=1:K,
            d(n,k) = norm(X(:,n)-mu(:,k));
        end
    end

    [~, ind] = min(d, [], 2);

    for n=1:N,
        k = ind(n);
        pi(k) = pi(k) + 1;
    end
    pi = pi./N;
elseif initParamsMethod == 1,
    %uniformly initialize the parameters
    pi = ones(K,1)*1./K;
    lmu = 0.25;
    umu = 0.75;
    mu = rand(D,K)*umu+lmu;
elseif initParamsMethod == 2,
%kmeans to initialize parameters
    IDX = kmeans(X', K);
    mu = zeros(D, K);
    pi = zeros(K, 1);
    for k=1:K,
        idxofk = find(IDX==k);
        X_k = X(:,idxofk);
        mu(:,k) = mean(X_k, 2);
        pi(k) = length(idxofk)./N;
    end
end

end

function p=computerBernProb(x, mu)
%computer Bernoulli pdf for sample x
D = length(x);
p=1;
for d=1:D,
    p = p * mu(d)^x(d)*(1-mu(d))^(1-x(d));
end
end

function f = computerObjVal(X, K, pi, mu)
%computer the value of log likelihood function
N = size(X,1);
f = 0;
for n=1:N,
    s=0;
    for k=1:K,
        s = s + pi(k) * computerBernProb(X(:,n),mu(:,k));
    end
    f = f + log(s);
end
end




    