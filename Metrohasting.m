close all; clc; clear variables;

%% Data generating process
N = 10000;
X = [ones(N,1) rand(N,1)]; % Data
BETA = [2 2]';
Y = X*BETA + randn(N,1);

%% Fit linear regression
betals = X\Y;
E = Y-X*betals;
sigma = var(E,0);
EstCov = sigma*(X'*X)\eye(2,2);

mdl = fitlm(X(:,2),Y);
%% Metropolis-Hastings

% Prior distribution
Beta0 = [2 2]';
Sigma0 = [0.1 0; 0 0.1];

% Likelihood

Sigma1 = ((1/sigma)*(X'*X)+Sigma0\eye(2))\eye(2);
Beta1 = Sigma1*((1/sigma)*(X'*Y)+(Sigma0\eye(2))*Beta0);

betai = [2 2]'; % initial guess

Nmax = 50000; % iteration
betastore = NaN(Nmax,2); % Matrix to store the value of betai

for i = 1:Nmax
    betastore(i,:) = betai';
    betastar = mvnrnd(Beta0,Sigma0)';
    
    p = @(b,sigma) exp(-0.5*(b - Beta1)'*((1/sigma)*(X'*X)+Sigma0\eye(2))*(b - Beta1)); % likelihood function
    q = @(b) exp(-0.5*(b - Beta0)'*(Sigma0\eye(2))*(b - Beta0)); % proposal distribution
    
    rho = min((p(betastar,sigma)/p(betai,sigma))*(q(betai)/q(betastar)),1);
    if rand() < rho % generate random uniform random variable
        betai = betastar;
    end
   
end

figure;plot(betastore(:,1)); title('constant');
figure;plot(betastore(:,2)); title('coefficient');

figure;histogram(betastore(:,1));
scatter(betastore(:,1),betastore(:,2));