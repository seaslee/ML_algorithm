function K_opt=train_model(X,Y,fold,K_min,K_max)
% K_opt=train_model(X,fold,K_min,K_max) computer the optimised k for k-NN.
% args:
%   X is the training data
%   Y is the labels for the training data
%   fold is the n number for the n-fold validation
%   K_min is the minimum number 
%   K_max is the maximum number
% return:
%   K_opt is the optimised value for k-NN
n=size(X,2);
foldnum=n/fold;
best_acc=0;
for k=K_min:K_max,
    acc=[];
    for i=1:fold,
        % put the data in part for n-fold validation
        test_index=1+(i-1)*foldnum:i*foldnum;
        if i==1,
            train_index=i*foldnum+1:n;
        elseif i==fold,
            train_index=1:(i-1)*foldnum;
        else
            train_index=[1+(i-2)*foldnum:(i-1)*foldnum,1+i*foldnum:n];
        end
        X_train=X(:,train_index);
        Y_train=Y(train_index);
        X_test=X(:,test_index);
        Y_test=Y(test_index);
        % test k on validation set 
        test_nums=size(X_test,2);
        Y_pr=[];
        for i=1:test_nums,
            x=X_test(:,i);
            kindex=knn(X_train,x,k);
            labels=Y_train(kindex);
            label=mode(labels);
            Y_pr=[Y_pr;label];
        end
        a=sum(Y_test==Y_pr)/size(Y_test,1);
        acc=[acc a];
    end
    % select the best k
    macc=mean(acc);
    if k=K_min,
        bestacc=macc;
        K_opt=k;
    elseif macc>bestacc
        bestacc=macc;
        K_opt=k;
    end
end

