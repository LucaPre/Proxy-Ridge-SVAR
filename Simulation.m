clear
rng(1)
MC=500; % Number of simulations
h=20; % Impulse response horizon
MSE=zeros(MC,3,3,h); % Space to store Monte Carlo results
A=[0.74 -0.09 -0.16; 0.13 0.44 -0.06; 0.24 0.30 0.53]; 
B=[2.32 -0.48 -0.41; 0.72 2.32 -0.22; 0.98 1.57 0.76];
irftrue=irfs([zeros(3,1) A],B,h); % True impulse response functions
T=270; % Sample size
lambda=log(T)/T; % Overall shrinkage term
instrument_variance=0.0021; % Estimated variance in instrument regression (empirical estimate from data application)
burn=100; % Burn in sample of DGP
for m=1:MC

y=zeros(T+burn,3);
shocks=zeros(T+burn,3);
instrument=zeros(T+burn,1);

% DGP 
for t=2:T+burn
    shocks(t,:)=randn(3,1)';
    y(t,:)=(A*y(t-1,:)'+B*shocks(t,:)')';
    instrument(t)=-0.0134+0.0142*shocks(t,3)+randn(1)*sqrt(instrument_variance); % instrument equation from empirical estimates
end
y=y(burn:end,:);

% Reduced Form
p=1;
instrument=instrument(burn+p:end);
[Ahat, yeff, x]=ReducedVAR(y,p); 
residuals=yeff-x*Ahat'; 
Sigma_U=cov(residuals);

% Cholesky
B_chol=chol(Sigma_U,"lower");

% Proxy GMM
poolobj = gcp('nocreate'); % Check if a parallel pool is already open
if isempty(poolobj)
    parpool; % Open a parallel pool if none exists
end
x0=B;
Aineq=[];
Bineq=[];
Aeq=[];
Beq=[];
lb=[];
ub=[];
opts = optimoptions(@fmincon,'Algorithm','interior-point','MaxFunctionEvaluations',1000,'Display','off');
ms = MultiStart('UseParallel',true,'Display','off','XTolerance',0.0000001,'FunctionTolerance',0.0000001);
fixedFunction = @(x) GMM_IV(x,instrument,residuals,eye(2),3);
fixedConstraint = @(x) nonlconivgmm(x,residuals,3);
problem = createOptimProblem('fmincon','x0',x0,'objective',fixedFunction,'nonlcon',fixedConstraint,'options',opts);
parfevalOnAll(@() warning('off', 'MATLAB:nearlySingularMatrix'), 0);
[Bhat_Proxy,fval,exitflag] = run(ms,problem,4);

% Proxy Ridge GMM
v=1./((Bhat_Proxy-B_chol).^2); 
v(:,1:2)=0;
v(3,3)=0;
poolobj = gcp('nocreate'); % Check if a parallel pool is already open
if isempty(poolobj)
    parpool; % Open a parallel pool if none exists
end
x0=Bhat_Proxy;
Aineq=[];
Bineq=[];
Aeq=[];
Beq=[];
lb=[];
ub=[];
opts = optimoptions(@fmincon,'Algorithm','interior-point','MaxFunctionEvaluations',1000,'Display','off');
ms = MultiStart('UseParallel',true,'Display','off','XTolerance',0.001,'FunctionTolerance',0.001);
fixedFunction = @(x) GMM_IV_Ridge(x,instrument,residuals,(0.0142^2+instrument_variance)^-1*eye(2),lambda,B_chol,v,3);
fixedConstraint = @(x) nonlconivgmm(x,residuals,3);
problem = createOptimProblem('fmincon','x0',x0,'objective',fixedFunction,'nonlcon',fixedConstraint,'options',opts);
parfevalOnAll(@() warning('off', 'MATLAB:nearlySingularMatrix'), 0);
[Bhat_Proxy_Ridge,fval,exitflag] = run(ms,problem,4);

% impulse responses of all strategies
irf_chol=irfs(Ahat,B_chol,h);
irf_proxy=irfs(Ahat,Bhat_Proxy,h);
irf_ridge=irfs(Ahat,Bhat_Proxy_Ridge,h);
for i=1:3
    for j=1:h
MSE(m,i,1,j)=(irf_chol(i,3,j)-irftrue(i,3,j))^2;
MSE(m,i,2,j)=(irf_proxy(i,3,j)-irftrue(i,3,j))^2;
MSE(m,i,3,j)=(irf_ridge(i,3,j)-irftrue(i,3,j))^2;
    end
end
clc
m
end

titles={'MP Shock on Output Gap', 'MP shock on Inflation', 'MP shock on Interest Rate'};
figure
for i=1:3
    subplot(1,3,i)
    plot(squeeze(mean(MSE(:,i,1,1:12))),'DisplayName','Cholesky')
    title(titles{i})
    if i==1
    ylabel('MSE','FontSize',12)
    end
    xlabel('horizon')
    hold on
    plot(squeeze(mean(MSE(:,i,2,1:12))),'DisplayName','Proxy')
    hold on
    plot(squeeze(mean(MSE(:,i,3,1:12))),'DisplayName','Ridge','Color','g')
    legend('Location','best')
end




