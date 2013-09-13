
function d = sigmoid_grad(x)
y=sigmoid(x);
d=y.*(1-y);
end
