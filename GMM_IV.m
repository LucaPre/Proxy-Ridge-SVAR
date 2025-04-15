function Q = GMM_IV(Bhat,instrument,residuals,w,shockpos)
k=length(Bhat);
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
Q=Moments*w*Moments';
end