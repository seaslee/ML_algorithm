%=============================================================================
%     FileName: getImage.m
%         Desc: 
%       Author: XuXinchao
%        Email: xxinchao@gmail.com
%     HomePage: http://webdancer.is-programmer.com
%      Version: 0.0.1
%   LastChange: 2012-11-14 16:35:51
%      History:
%=============================================================================

function [X,Y]=getImage(files)
% the function is to put the data set into the data matrix and labels into a vector
%
% args: 
%   files is the path of the data set
% return : 
%   X is a matrix which one column is one training instance.
%   Y is the label vector.
%
% the directory structure is like following:
% data---|
%        |-S1
%           |-images
%        |-S2
%           |-images
%        |-...

%get the subdirs 
subdirs=listFiles(files);
%put the image into data matrix Y and give a label for every face category
n=size(subdirs,1);
X=[];
Y=[];
for i=1:n,
    imgdir=subdirs{i};
    images=listFiles([files filesep imgdir])
    img_nums=size(images,1)
    for j=1:img_nums,
        j
        img=imread([files filesep imgdir filesep images{j}]);
        [r,c]=size(img);
        img=double(img);
        img=reshape(img,r*c,1);
        X=[X img];
        Y=[Y;i];
    end
end
end

function f=listFiles(files)
%get the subdirs and files
f=dir(files);
f=f(3:end,:);
f=struct2cell(f);
f=f(1,:);
f=f';
end

