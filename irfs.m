% returns matrices of impulse responses for given reduced form and
% structural parameter estimates and horizons
function IRFs = irfs(Ahat,Bhat,h)

k=length(Bhat);
p=(width(Ahat)-1)/k;
C=zeros(p*k,p*k); % Generate companion form
C(1:k,:)=Ahat(:,2:end);
for i=1:p-1
    C(i*k+1:i*k+k,(i-1)*k+1:i*k)=eye(k);
end

IRFs=zeros(k,k,h);
for i=1:h
    Lags=C^(i-1);
    IRFs(:,:,i)=Lags(1:k,1:k)*Bhat;
end
end


