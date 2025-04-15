% Estimates A from reduced VAR yt=AXt'+ut by OLS (Xt=[1 yt-1 yt-2...yt-p])
% Also returns observation matrix y and X after dropping the required presample
% for estimation and further use
function [Ahat,y,x]=ReducedVAR(y,p)

x=getLags(y,p);
T=size(y,1);
x= [ones(T-p,1) x];
y=y(1+p:end,:);

Ahat=zeros(width(y),width(x))';
for i=1:width(y)
Ahat(:,i)=(x'*x)^-1*x'*y(:,i); 
end

Ahat=Ahat';
end