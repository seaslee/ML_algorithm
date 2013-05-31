function  [denMat, iterNum] = agglcluster(X, varargin)
% agglomerative clustering
% Parameters:
%     -X: dataset matrix whose column is sample and row is feature
% Options:
%     -metric: used to computer the similarity matrix 
%     -intergroupdist: 0: single linkage, 1: max linkage, 2: group average
% Return:
%     -dendrogram: the tree of clusters
%     -iterNum: the iteration number of algorithm

s.metric = 'euclidean';
s.intergroupdist = 2;
if ~isempty(varargin),
     s = handleoptions(s,varargin);
end

N = size(X, 2)
D = pdist(X', s.metric);
D = squareform(D);

iterNum = 1; %step
denMat = zeros(N-1, 3);
label = 1:N;

while iterNum < N,
    y = computerMinInterClusterDist();
    if y(1)>N,
        indx1 = findLeaves(y(1));
    else
        indx1 = y(1);
    end
    if y(2)>N,
        indx2 = findLeaves(y(2));
    else
        indx2 = y(2);
    end
    label(indx1) = N + iterNum;
    label(indx2) = N + iterNum;
    denMat(iterNum, :) = y;
    iterNum = iterNum + 1;
    %fprintf('Iteration number: %d, indx1: %d, indx2: %d\n',iterNum,indx1,indx2);
end

function  y=computerMinInterClusterDist()
clusterlabels = unique(label);  
clusterNum = length(clusterlabels);
fprintf('Check: iterative number: %d ; cluster number is %d\n',iterNum, clusterNum);
mininterdist = inf;
for i=1:clusterNum,
    for j=i+1:clusterNum,
        x1 = find(label==clusterlabels(i));
        x2 = find(label==clusterlabels(j));
        N_x1 = length(x1);
        N_x2 = length(x2);
        if s.intergroupdist ==0,
            minintradist = inf;
            for m=1:N_x1,
                for n=1:N_X2,
                    d = D(x1(m),x2(n));
                    if d < minintradist,
                        minintradist = d;
                    end
                end
            end
            interdist = minintradist;
        elseif s.intergroupdist == 1,
            maxintradist = -inf;
            for m=1:N_x1,
                for n=1:N_x2,
                    d = D(x1(m),x2(n));
                    if d > maxintradist,
                        maxintradist = d;
                    end
                end
            end
            interdist = maxintradist;
        elseif s.intergroupdist == 2,
            sumintradist = 0;
            for m=1:N_x1,
                for n=1:N_x2,
                    d = D(x1(m),x2(n));
                    sumintradist = sumintradist + d;
                end
            end
            interdist = sumintradist/(N_x1*N_x2);
        else
            error('wrong parameter intergroupdist value %d',s.intergroupdist);
        end
        if interdist < mininterdist,
            mininterdist = interdist;
            y(1) = clusterlabels(i);
            y(2) = clusterlabels(j);
            y(3) = mininterdist;
        end
    end
end
end

function y=findLeaves(clusterNum)
%find leaves need to change label
queue = [clusterNum];
y = [];
while ~isempty(queue),
     nl = queue(1);
     queue(1) = [];
     ch = denMat(rem(nl,N),1:2);
     if ch(1)<=N,
         y = [y ch(1)];
     else
         queue = [queue  ch(1)];
     end
     if ch(2)<=N,
         y = [y ch(2)];
     else
         queue = [queue  ch(2)];
     end
end

end

end




