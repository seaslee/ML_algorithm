 function loss = eval_val(X_val,Y_val,W,layernums,active_fun)
% Data: X_train, Y_train. add the (1,1,1,...,1)' to X's raw input
% parameter: w1 w2 is the parameter for first and second layer
% eta : the learning rate
% W = model.weights;
% layernums = model.layers;
z{1} = X_val;
a{1} = X_val;
for i=1:layernums,
    a{i} = [ones(1,size(a{i},2)); a{i}];
    z{i+1} = W{i} * a{i};
    a{i+1} = active_fun(z{i+1});
end
Y_new = a{i+1};
[D,N]=size(Y_new);
Y_pred = zeros(D, N);
for i=1:N,
    [y,ind]=max(Y_new(:,i));
    Y_pred(ind,i) = 1;
    for j=1:D,
        if j==ind,
            Y_new(j,i)=1;
        else
            Y_new(j,i)=0;
        end
    end
end

pre_wrong=0;
pre_nums=size(Y_val,2);
for i=1:pre_nums,
    if ~isequal(Y_new(:,i),Y_val(:,i)),
        pre_wrong=pre_wrong+1;
    end
end
loss=pre_wrong/pre_nums;
% fprintf('the loss is %f\n',loss);

end


