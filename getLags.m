% Gets matrix x of length T-p which contains the first p Lags of a (T x k) data vector y
function x=getLags(y,p)
[T k]=size(y);
x=zeros(T-p,p*k);
count=1;
for i= 1:p
    for j=1:k
        x(:,count)=[y(p+1-i:end-i,j)]; 
        count=count+1;
    end
end
end