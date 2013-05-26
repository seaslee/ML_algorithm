function [model, converged, iterNum, optObjVal, r] = gmm(X, K, threshold, maxIterNum, initParamsMethod)
% MLE for GMM with EM algorithm
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

[pi, mu, Sigma] = initParams(X, K, initParamsMethod);

[D, N] = size(X);
converged = 0;
iterNum = 0;
r = zeros(N,K);
preObjVal = computerObjVal(X, K, pi, mu, Sigma);

%EM iteration
while iterNum < maxIterNum,
    iterNum = iterNum + 1;
    %E step: computer responsibility of k component for nth sample
    for n=1:N,
        for k=1:K,
            r(n,k) = pi(k) * mvnpdf(X(:,n), mu(:,k), Sigma(:,:,k));
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
    for k=1:K,
        sigma = zeros(D);
        for n=1:N,
            sigma = sigma + r(n,k)*(X(:,n)-mu(:,k))*(X(:,n)-mu(:,k))';
        end
        if det(sigma)==0,
            fprintf('fuck the conv matrix\n');
        end
        Sigma(:,:,k) = sigma./Nk(k);
    end
    
    curObjVal = computerObjVal(X, K, pi, mu, Sigma);
    
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
model.Sigma = Sigma;
optObjVal = curObjVal;

end

function [pi, mu, Sigma] = initParams(X, K, initParamsMethod)
% initialize paramters 
    [D,N] = size(X);
if initParamsMethod ==0,
    %randomly initialize parameter
    perm = randperm(N);
    mu = X(:,perm(1:K));
    pi = zeros(K, 1);
    Sigma = zeros(D, D, K);

    d = zeros(N,K);
    for n=1:N,
        for k=1:K,
            d(n,k) = norm(X(:,n)-mu(:,k));
        end
    end

    [~, ind] = min(d, [], 2);

    for n=1:N,
        k = ind(n);
        Sigma(:,:,k) = Sigma(:,:,k) + (X(:,n)-mu(:,k)) * (X(:,n)-mu(:,k))';
        pi(k) = pi(k) + 1;
    end
    pi = pi./N;
    
elseif initParamsMethod == 1,
%kmeans to initialize parameters
    IDX = kmeans(X', K);
    mu = zeros(D, K);
    pi = zeros(K, 1);
    Sigma = zeros(D, D, K);
    for k=1:K,
        idxofk = find(IDX==k);
        X_k = X(:,idxofk);
        mu(:,k) = mean(X_k, 2);
        pi(k) = length(idxofk)./N;
        Sigma(:,:,k) = cov(X');
    end
end

end


function f = computerObjVal(X, K, pi, mu, Sigma)
%computer the value of log likelihood function
N = size(X,1);
f = 0;
for n=1:N,
    s=0;
    for k=1:K,
        s = s + pi(k) * mvnpdf(X(:,n),mu(:,k),Sigma(:,:,k));
    end
    f = f + log(s);
end
end


    
