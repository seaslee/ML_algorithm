function model = dbn(X,layernums,hidnums,varargin)
% train the parameters in DBN models
% Parameters:
% X: D*N data matrix
% layernums: the layer numbers of hidden layers
% hidnums: the numbers in each hidden layer
% Return:
% models: struct to store DBN model

%prepare options 
s.maxiternum = 100;
s.rate = 0.01;
s.batchsize = 100;
s.initmomentum = 0.5;
s.finalmomentum = 0.9;
if ~isempty(varargin),
    s = handleoptions(s,varargin);
end

[D,N] = size(X);

%% ======pre-training dbn by stacking RBM=====
for i=1:layernums,
    rbm_model = rbm(X,hidnums(i));
    rbms{i}=rbm_model;
    X = rbm_hid_stats(X,rbm_model,N);
end

model.layernums = layernums;
model.hidnums = hidnums;
model.rbms = rbms;

end

function h = rbm_hid_stats(X,model,data_n)
W = model.weight;
hbias = model.hbias;
hidnum = model.hidnum;
p_h = sigmoid(bsxfun(@plus, W'*X, hbias));
h = p_h > rand(hidnum, data_n);
end

