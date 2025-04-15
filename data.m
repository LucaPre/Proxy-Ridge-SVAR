clear
data_table = readtable('GKdata.csv');  % load data
col_names = ["logip" "logcpi" "gs1" "ebp" "ff4_tc"];
variable_names = ["IP" "CPI" "One-year rate" "Excess bond premium"];
instrument = data_table(:,"ff4_tc"); % select instrument
instrument = table2array(instrument);
plag = 12; % number of lags
instrument = instrument((plag+1):size(data_table,1),:);
dependent = data_table(:,"gs1"); % select dependent variable
data_table = data_table(:,col_names(1:end-1));
y = table2array(data_table);
y=y(115:end,:);
instrument=instrument(115:end);

% Reduced Form
p=plag;
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
x0=[B_chol(:,1) B_chol(:,2) B_chol(:,4) -B_chol(:,3)];
Aineq=[];
Bineq=[];
Aeq=[];
Beq=[];
lb=[];
ub=[];
opts = optimoptions(@fmincon,'Algorithm','interior-point','MaxFunctionEvaluations',1000,'Display','off');
ms = MultiStart('UseParallel',true,'Display','off','XTolerance',0.001,'FunctionTolerance',0.001);
fixedFunction = @(x) GMM_IV(x,instrument,residuals,eye(3),3);
fixedConstraint = @(x) nonlconivgmm(x,residuals,3);
problem = createOptimProblem('fmincon','x0',x0,'objective',fixedFunction,'nonlcon',fixedConstraint,'options',opts);
parfevalOnAll(@() warning('off', 'MATLAB:nearlySingularMatrix'), 0);
[Bhat_Proxy,fval,exitflag] = run(ms,problem,4);

% Regression of instrument on MP shock for simulation calibration
shocks=(Bhat_Proxy^-1*residuals')';
MPshock=shocks(:,4);
X=[ones(270,1) MPshock];
betahat=(X'*X)^-1*X'*instrument;
variance=(instrument-X*betahat)'*(instrument-X*betahat)/270;

% Proxy Ridge GMM
B0=B_chol;
v=1./((Bhat_Proxy-B0).^2);
v(:,1:2)=0;
v(3:4,3)=0;
v([1;2;4],4)=0;
poolobj = gcp('nocreate'); % Check if a parallel pool is already open
if isempty(poolobj)
    parpool; % Open a parallel pool if none exists
end
x0=B_chol;
Aineq=[];
Bineq=[];
Aeq=[];
Beq=[];
lb=[];
ub=[];
opts = optimoptions(@fmincon,'Algorithm','interior-point','MaxFunctionEvaluations',1000,'Display','off');
ms = MultiStart('UseParallel',true,'Display','off','XTolerance',0.001,'FunctionTolerance',0.001);
fixedFunction = @(x) GMM_IV_Ridge(x,instrument,residuals,var(instrument)^-1*eye(3),log(270)/270,B_chol,v,3);
fixedConstraint = @(x) nonlconivgmm(x,residuals,3);
problem = createOptimProblem('fmincon','x0',x0,'objective',fixedFunction,'nonlcon',fixedConstraint,'options',opts);
parfevalOnAll(@() warning('off', 'MATLAB:nearlySingularMatrix'), 0);
[Bhat_Proxy_Ridge,fval,exitflag] = run(ms,problem,4);

h=36;
irf_chol=irfs(Ahat,B_chol,h);
irf_proxy=irfs(Ahat,Bhat_Proxy,h);
irf_ridge=irfs(Ahat,Bhat_Proxy_Ridge,h);

figure
for i=1:4
    subplot(2,2,i)
    plot(squeeze(irf_chol(i,3,:)))
    xlabel('horizon')
    ylabel('Response')
    title(variable_names{i})
end

figure
for i=1:4
    subplot(2,2,i)
    plot(squeeze(irf_proxy(i,3,:)))
    xlabel('horizon')
    ylabel('Response')
    title(variable_names{i})
end

figure
for i=1:4
    subplot(2,2,i)
    plot(squeeze(irf_ridge(i,3,:)))
    xlabel('horizon')
    ylabel('Response')
    title(variable_names{i})
end