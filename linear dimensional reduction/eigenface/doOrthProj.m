%%This is simple script to do orthogonal projection in 2-dimesion space

function  doOrthProj(X,u)

%plot data
x1=X(1,:);
x2=X(2,:);
plot(x1,x2,'r.','MarkerSize',15);
axis([0,round(max(x1))+5,0,round(max(x2))+5]);
xlabel('x1');
ylabel('x2');
hold on;
% %plot the line
slope=u(2)/u(1);
x=0:round(max(x1))+5;
plot(x,slope*x,'b','LineWidth',2);
% hold on;
%plot orthogonal projection vector
u=u/norm(u);
p=u*u';
y=p*X;
%plot the projected data
plot(y(1,:),y(2,:),'k.','MarkerSize',15);
hold on;
% plot the projected line
n=size(X,2);
for i=1:n,
    px=[X(1,i),y(1,i)];
    py=[X(2,i),y(2,i)];
    plot(px,py,'g--','LineWidth',2);
end;
end
