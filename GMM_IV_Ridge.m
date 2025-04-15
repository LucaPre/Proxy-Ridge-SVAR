function Q = GMM_IV_Ridge(Bhat,instrument,residuals,w,lambda,B_prior,v,shockpos)
k=length(Bhat);
T=size(residuals,1);
Moments=zeros(1,k-1);
shocks=(Bhat^-1*residuals')';
count=0;
for i=1:k
    if i==shockpos
        [];
    else
M=mean(shocks(:,i).*instrument);
count=count+1;
Moments(count)=M;
    end
end
Q=Moments*w*Moments'+lambda*sum(sum(v.*(Bhat-B_prior).^2));
end