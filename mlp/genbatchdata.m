function [batchdata,batchy,batchnum] = genbatchdata(X,Y,batchsize)
%  generate batchdata
[D, N] = size(X);
C = size(Y,1);
randp = randperm(N);
batchnum = ceil(N/batchsize);
batchdata = zeros(D, batchsize, batchnum);
batchy = zeros(C, batchsize, batchnum);
for i=1:batchnum,
    if i ~= batchnum,
        index = randp(1+(i-1)*batchsize:i*batchsize);
    else
        index = randp(1+(i-1)*batchsize:N);
    end
    batchdata(:,:,i) = X(:,index);
    batchy(:,:,i) = Y(:,index);
end
end