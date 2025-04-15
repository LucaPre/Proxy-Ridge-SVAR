function [c, ceq] = nonlconivgmm(x0,residuals,shockpos)
Sigma_U=cov(residuals);
res=(x0*x0'-Sigma_U);
logmat=triu(true(size(x0)), 1);
logmat(:,shockpos)=0;
logmat(shockpos,:)=0;
ceq=[res(:); x0(logmat)];
c=[-diag(x0)];
end