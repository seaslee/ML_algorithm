function model = mlp(X_train,Y_train,layernums,hidnums,outnums,varargin)
% use stochastic gradient descent to train parameter of mlp
% Parameters:
%   -X_train: D*N training dataset
%   -Y_train: labels of training dataset
%   -layers: layer numbers of network
%   -hidnums: hidden uints number for each hidden layer
%   -outnums: output units number
% Returns:
%   -model: struct for mlp model
% ==============================
% Author: XuXinchao
% Email: xxinchao@gmail.com
% HomePage: http://webdancer.is-programmer.com

% ==== handle options ====
s.epochs = 100;
s.rate = 0.2;  %learning rate
s.ratedec = 0.85;
s.batchsize = 100;
s.initmomentum = 0.5;  %momentum parameter
s.finalmomentum = 0.99;
s.stathreshold = 100;
s.epsilon = 1e-10;
s.active_fun=@sigmoid;   %active function 
s.active_fun_grad=@sigmoid_grad;  
s.lambda = 0.01; %regularization parameter
s.patience = 2000; % early stopping paramter
s.patienceinc = 2;
s.improvethres = 0.99;
s.checkval = 0;
s.dataval = 0;
s.targetval = 0;
s.dropout = 0; %dropout parameter
s.visualdroprate = 0.5;
s.hiddendroprate = 0.5;
s.inweighthreshold = 15;
if ~isempty(varargin),
    s = handleoptions(s,varargin);
end
% ====== get variables from paramters ======
epochs = s.epochs;
rate = s.rate ;
ratedec = s.ratedec;
batchsize = s.batchsize;
% generate the mini-batch data
size(X_train)
[batchdata, batchy, batchnum] = genbatchdata(X_train,Y_train, batchsize);
%===
momentum = s.initmomentum;
stathreshold = s.stathreshold ;
epsilon = s.epsilon;
active_fun= s.active_fun;
active_fun_grad = s.active_fun_grad;
lambda = s.lambda;
patience = s.patience ; % early stopping paramter
patienceinc = s.patienceinc ;
improvethres = s.improvethres ;
valfreq = round(min(batchnum, patience/2));
checkval = s.checkval;
if checkval,
    X_val = s.dataval;
    Y_val = s.targetval;
end
dropout = s.dropout; %dropout
visualdroprate = s.visualdroprate;
hiddendroprate = s.hiddendroprate;
inweighthreshold = s.inweighthreshold;
[D, ~] = size(X_train);

% % ===pre-training weights for NN using DBN====
% model = dbn(X_train,layernums,hidnums);
% for i=1:layernums-1,
%     W{i} = [model.rbms{i}.hbias,model.rbms{i}.weights'];
% end
% lastinputnums = model.rbms{layernums}.hbias;
% epsilon_init = sqrt(6)./sqrt(lastinputnums + outnums)
% W{layernums} = rand(lastinputnums, outnums)*2*epsilon_init-epsilon_init;

% ===initialize the weights of NN====
unitnums = [D hidnums outnums];
for i=1:layernums,
    inputnums = unitnums(i) + 1;
    outputnums = unitnums(i+1);
    epsilon_init = 4*sqrt(6)/sqrt(inputnums + outputnums);
    W{i} = rand(inputnums, outputnums)* 2 * epsilon_init - epsilon_init;
    W{i} = W{i}';
    W_inc{i} = zeros(outputnums, inputnums);
end

% ===Stochastic gradient descent========
cost_fun_vals=[];

loopfinshed = 0;

bestvalloss = inf;    
batchcost = 0;
valloss = 0;
while i < epochs && ~loopfinshed,
    randbatch = randperm(batchnum);
    batchdata = batchdata(:,:,randbatch);
    batchy = batchy(:,:,randbatch);
    for n=1:batchnum,
        [W_grad, cost] = backprop(batchdata(:,:,n),batchy(:,:,n),...
                                  layernums,W,active_fun,...
                                  active_fun_grad,'dropout',dropout,...
                                  'hiddendroprate',hiddendroprate);
        batchcost = batchcost + cost;
        iter = (i-1)*batchnum + n;
        for k=1:layernums,
            W_grad{k}(:,2:end) = W_grad{k}(:,2:end) + lambda * W{k}(:,2:end);
            W_inc {k} = momentum * W_inc {k} - rate * W_grad{k};
            W{k} = W{k} + W_inc {k};
            %dropout
            if dropout,
%                 fprintf('fuck\n');
                inweightlen = sqrt(sum(bsxfun(@times,W{k}(:,2:end),W{k}(:,2:end))));
                mask = inweightlen > inweighthreshold;
                dividor = inweightlen .* mask + (~mask);
                W{k}(:,2:end) = bsxfun(@rdivide, W{k}(:,2:end), dividor);
            end
            
        end
        
        if mod(iter, valfreq) == 0,
            if checkval,
                valloss = eval_val(X_val,Y_val,W,layernums,active_fun);
                if valloss < bestvalloss,
                    bestvalloss = valloss;
                    bestw = W;
                    if valloss < bestvalloss * improvethres,
                        patience = max(patience, iter * 2);
                    end
                end
            end
            batchcost = batchcost/valfreq;
            cost_fun_vals = [cost_fun_vals batchcost];
            fprintf('Iteration: %d, train cost is %f , val cost is %f\n',iter, batchcost, valloss);
            batchcost = 0;
        end
        
        if patience < iter && checkval,
            loopfinshed = 1;
        end
        
%         if mod(n,5)==0,
%              batchcost = batchcost./batchnum;
%              cost_fun_vals = [cost_fun_vals batchcost];
%              fprintf('Iteration: %d, the cost is %f \n',iter, batchcost);
%              batchcost = 0;
%         end
    end
    
    if i > stathreshold,
        momentum = s.finalmomentum;
    else
        pp = i/stathreshold;
        momentum = pp*s.initmomentum + (1- pp)*s.finalmomentum;
    end
    
    rate = rate * ratedec;
    i = i+1;
%     rate = rate /(1+ ratedec * i);
end

if ~checkval,
    bestw = W;
end
model.layers = layernums;
model.hidnums = hidnums;
model.outnums = outnums;
model.weights = bestw;
model.cost = cost_fun_vals;
model.valloss = bestvalloss;
model.dropout = dropout;
model.visualdroprate = visualdroprate;
model.hiddendroprate = hiddendroprate;

end
