function  Y=RBMGenData(model, X, K)
% Generate data from RBM model
% Parameters:
%   -model: RBM model struct
%   -X: input
%   -K: steps for gibbs sampling
N = size(X, 2);
w = model.weight;
vbias = model.vbias;
hbias = model.hbias;
Y=[];
for n=1:N,
    v=X(:,n);
    for k=1:K,
        h = sigmoid(w'* v + hbias);
        v = sigmoid(w * h + vbias);
    end
    Y=[Y v];
end
end

function y=sigmoid(X)
y=1./(1+exp(-X));
end
    