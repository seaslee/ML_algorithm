close all;
clear;
clc;

load data;

layers = [2];
hidnumsrange = 300:100:800;
outnums = 10;
lambda = -4:1:2;
lr = [0.05,0.1,1,3,5,7,9];
lrdec = [0.9,0.8,0.7,0.6,0.5,0.4];
batchsize = [10,50,100];
% 
% active_fun=@sigmoid;
% active_fun_grad=@sigmoid_grad;
% grad = checkgradwithfinitediff(X_train,Y_train,layers,hidnums,outnums,active_fun,active_fun_grad)
[fid, mes] = fopen('log.txt', 'w+');
bestloss = inf;
for layind = 1:length(layers),
    for hidind = 1:length(hidnumsrange),
        for lamind = 1:length(lambda),
            for lrind = 1:length(lr),
                for lrdecind = 1:length(lrdec),
                    for batchind = 1:length(batchsize),
                        lay = layers(layind);
                        hn = hidnumsrange(hidind);
                        lam = lambda(lamind);
                        lre = lr(lrind);
                        lrdec = lrdec(lrdecind);
                        batchs = batchsize(batchind);
                        model = mlp(traindata,traintarget,lay,hn,outnums,...
                                    'lambda',lam,'checkval',1,'rate',lre,...
                                    'ratedec',lrdec,'batchsize',batchs,...
                                    'dataval',valdata,'targetval',valtarget);
                        currloss = model.valloss;
                        if currloss < bestloss,
                            bestloss = currloss;
                            bestmodel = model;
                            fprintf(fid,'============================\n');
                            fprintf(fid,'hidden: %d, lambda: %f, learningrate: %f, ratedecr: %f, batch: %d, valloss: %f\n', hn,lam,lre,lrdec,batchs,currloss);
                            fprintf(fid,'============================\n');
                        end
                    end
                end
            end
        end
    end
end
fclose(fid);
save bestmodel bestmodel;
[Y_new1, acc1] = predict(traindata,traintarget,bestmodel,@sigmoid);
[Y_new2, acc2] = predict(testdata,testtarget,bestmodel,@sigmoid); 