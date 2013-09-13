function [model, iternum] = rbm(X, hidnum, varargin)
% Perform the learning algorithm for RBM(Restriced Boltzmann machine)
% The learning algorithm is CD(contrastive divergence, hinton 2002)
% Parameters:
%   -X: d*n data matrix
%   -visnum: visual units number
%   -hidnum: hidden units number
% Options:
%   -maxiternum: maximum iterative number
%   -k: step of gibbs sampling for cd
%   -rate: learning rate
%   -batchsize: if 0, batch gradient descent is used; if 1, online; else
%               ,mini-batch is used and this is the batchsize(default)  
% Return:
%   -model: the rbm model struct
%   -converged: 1 if converged,otherwise, 0
%   -iternum: iterative number 
% ======================================
% Author: Xu Xinchao
% Blog: http://webdancer.is-programmer.com/
% ======================================

% ====handle options====
s.maxiternum = 100;
s.rate = 0.01;
s.batchsize = 100;
s.initmomentum = 0.5;
s.finalmomentum = 0.9;
if ~isempty(varargin),
    s = handleoptions(s,varargin);
end

maxiternum = s.maxiternum;
rate = s.rate;
batchsize = s.batchsize;
initmomentum = s.initmomentum;
finalmomentum = s.finalmomentum;
momentum=initmomentum;  % use for momentum methods

% ====initial weights=====
[D, N] = size(X);
visnum = D;
[W, vbias, hbias] = initweigsbias(X, visnum, hidnum);
W_ch = zeros(visnum,hidnum);
vbias_ch = zeros(visnum,1);
hbias_ch = zeros(hidnum,1);  
iternum = 0;
[batchdata, batchnum] = genbatchdata(X, batchsize);
while iternum <= maxiternum,
    sumerr = 0;
    if iternum>5,
        momentum = finalmomentum;
    end
    momentum
    for n = 1:batchnum,
        [grad_w, grad_v, grad_h, err] = cd(batchdata(:,:,n), W, vbias, hbias);
        
        W_ch = W_ch*momentum + rate*grad_w;
        vbias_ch = vbias_ch*momentum + rate*grad_v;
        hbias_ch = hbias_ch*momentum + rate*grad_h;  
        
        W = W + W_ch;
        vbias = vbias + vbias_ch;
        hbias = hbias + hbias_ch;
        
        sumerr = sumerr + err;
    end
    fprintf('Minibatch Iteration: %d, error: %f\n', iternum, sumerr);
    iternum = iternum + 1;
end

model.visnum=visnum;
model.hidnum=hidnum;
model.weight=W;
model.vbias=vbias;
model.hbias=hbias;

end
% ====batchdata generate====
function [batchdata, batchnum] = genbatchdata(X, batchsize)
[D, N] = size(X);
randp = randperm(N);
batchnum = ceil(N/batchsize);
batchdata = zeros(D, batchsize, batchnum);
for i=1:batchnum,
    batchdata(:,:,i) = X(:,randp(1+(i-1)*batchsize:i*batchsize));
end
end

% ====initialize weights for RBM====
function [W, vbias, hbias]=initweigsbias(X, visnum, hidnum)
% W: visnum * hidnum weights matrix
mu = 0;
sigma = 0.01;
% random sampling from gaussian disrit.
W = normrnd(mu,sigma,[visnum,hidnum]);
% bias for visual units
[data_n, feat_n] = size(X);
vbias = zeros(visnum,1);
% for i=1:visnum,
%     p = nnz(X(i,:))./data_n;
%     vbias(i) = p*(1-p);
% end
% bias for hidden units
hbias = zeros(hidnum,1);

end

% ====contrastive divergence algorithm====
function [grad_w, grad_v, grad_h, err] = cd(X, W, vbias, hbias)
data_n = size(X, 2);
hidnum = size(W, 2);

% ====positive phrase====
v0 = X;
p_h0 = sigmoid(bsxfun(@plus, W'*v0, hbias));
h0 = p_h0 > rand(hidnum, data_n);
posphr = v0 * p_h0'./data_n;
% ====end positive phrase===

% ====negative phrase====
v1 = sigmoid(bsxfun(@plus, W*h0, vbias));
p_h1 = sigmoid(bsxfun(@plus, W'*v1, hbias));
% h1 = p_h1 > rand(hidnum, data_n);
negphr = v1 * p_h1'./data_n;

%computer gradient
grad_w = posphr - negphr;
grad_v = mean(X,2)-mean(v1,2);
grad_h = mean(p_h0,2)-mean(p_h1,2);

%computer reconstruction error
err = sum(sum(X-v1));

end

function y=sigmoid(X)
y=1./(1+exp(-X));
end
    


    
