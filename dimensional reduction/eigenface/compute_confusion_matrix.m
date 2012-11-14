function [confusion_matrix]=compute_confusion_matrix(label,label_pr,num_class)

confusion_matrix=zeros(num_class,num_class);
C=[label label_pr];
n=size(C,1);
for ci=1:num_class
    for cj=1:num_class
	    for k=1:n,
            c=C(k,:);
		    if c(1)==ci && c(2)==cj,
			    confusion_matrix(ci,cj)=confusion_matrix(ci,cj)+1;
            end
        end
    end
end
for i=1:num_class,
	confusion_matrix(i,:)=confusion_matrix(i,:)/sum(confusion_matrix(i,:));
end

draw_cm(confusion_matrix,num_class);

end
