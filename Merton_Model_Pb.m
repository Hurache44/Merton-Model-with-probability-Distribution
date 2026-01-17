
%%%%
% Merton Model
%%%%

% Values are not correct
% Sharpe Ratio, MeanPrice, Vol., Calculations are not valid

% 1. Load data
% 2. Returns & market beta
% 3. Market Regime (SMA50/200)
% 4. Storage Initialization
% 5. Asset Loop
%       35 n_steps Regression
%       Regime best response
%       parameter estimation
%       OU-Merton simulation
%       target probability
% 6. summary tables
% 7. Rolling beta

% Additional Model
% Chatgpt code adoption after revising Merton_Model

%% =========
% Institutional Merton-OU Model with target probability
% Robert Lange % Chatgpt
%  =========

clear; clc; close all; rng(42);

% Setup
nPaths = 5000;
nYears = 2;
dt = 1/252;
steps = round(nYears/dt);

targetMultiplier = 1.15;
targetHorizonYears = 1;
targetStep = round(targetHorizonYears/dt);

regWindow = 35;
rf = 0.02;

% Data
cols = (2:5);
data = readtable('/Users/robertlange/Desktop/Investment/Matlab/Returns.xlsx');
dates = data{:,1};
prices = data{:,cols};
assetNames = data.Properties.VariableNames(cols);

[nObs,nAssets] = size(prices);

logP = log(prices);
rets = diff(logP);

RegY = cell(nAssets,1);
RegYHat = cell(nAssets,1);
SimEndPaths = cell(nAssets,1);
TargetPriceVec = nan(nAssets,1);

%%
% Market Return and Betas
marketRet = mean(rets,2,'omitnan');
betaVec = nan(nAssets,1);

for i = 1:nAssets
    r = rets(:,i);
    idx = ~isnan(r) & ~isnan(marketRet);
    ri = r(idx);
    rm = marketRet(idx);
    betaVec(i) = (ri-mean(ri))'*(rm-mean(rm))/((rm-mean(rm))'*(rm-mean(rm)));


    % betaVec(i) = cov(r(idx),marketRet(idx),1);
    % betaVec(i) = betaVec(i,2)/var(marketRet(idx));


end
%%
% Market Regime (SMA)
marketPrice = mean(prices,2,'omitnan');
SMA50 = movmean(marketPrice,[49 0]);
SMA200 = movmean(marketPrice,[199 0]);
    % -1 /  0 / +1
MarketRegime = sign(SMA50(end)-SMA200(end));

% Storage
SimMeanPrice    = nan(nAssets,1);
SimVol          = nan(nAssets,1);
SharpeSim       = nan(nAssets,1);

TargetProb      = nan(nAssets,1);
TargetProbHit   = nan(nAssets,1);
TargetHitTime   = nan(nAssets,1);

RegSlope        = nan(nAssets,1);
RegIntercept    = nan(nAssets,1);
RegR2           = nan(nAssets,1);
RegTargetPrice  = nan(nAssets,1);

SimPaths = cell(nAssets,1);
SimRets = cell(nAssets,1);

%%
% Asset Loop

for i = 1:nAssets
    price = prices(:,i);
    lp = log(price);
    lp = lp(~isnan(lp));
    % SimReturns{i} = diff(log(Paths),1,1);
    % last 35 regression (pre simulation)
if numel(lp) < regWindow
    continue
end
    y = lp(end-regWindow+1:end);
    t = (1:regWindow)';
    X = [ones(regWindow,1),t];
    coeff = X\y;

    n = coeff(1);  % intercept
    m = coeff(2);  % slope

    yHat = X*coeff;
    % R^2
    SSr  = sum((y-yHat).^2);
    SSt  = sum((y-mean(y)).^2);
    R2   = 1-SSr/SSt;

    RegIntercept(i) = n;
    RegSlope(i)     = m;
    RegR2(i)  = 1-SSr/SSt;
    % TG projection
    futureX = regWindow+targetStep;
    RegTargetPrice(i) = exp(m*futureX+n);

    % Parameter estimation
    % Original formula
    % params = estimate_OU_jump(lp,dt);
    % muGBM = RegSlope(i)*252;
    
    lambdaReg = 0.15*betaVec(i);
    
    % Return moments
    ret = diff(lp);
    mu = mean(ret,'omitnan')*252;
    sig = std(ret,'omitnan')*sqrt(252);
    mu = mean(ret,'omitnan')*252;

    muSignal = m*252;
    
    muAdj = mu+0.15*betaVec(i)*MarketRegime;
    signal = max(min(muSignal/(sig+eps),2),-2);
    position = 0.5+0.25*signal;
    sigmaGBM = std(diff(lp))*sqrt(252);
    
    params = estimate_jump_params(lp,dt);
    % muAdj = mu+0.15*betaVec(i)*MarketRegime;    

    % params.theta = params.theta+lambdaReg*MarketRegime;
    % Regime best response
    % muAdj   = mu+0.15*betaVec(i)*MarketRegime;
    sigAdj  = sig*(1+0.3*(MarketRegime==-1));

    % Simulation
    % steps   = nYears*252;
    % dt      = nYears/steps;
    S0 = price(find(~isnan(price),1,'last'));

    Paths = simulate_GBM_Merton( ...
        muAdj, ...
        sigAdj, ...
        params.lambdaJ, ...
        params.muJ, ...
        params.sigJ, ...
        S0,nYears,dt,nPaths);

    Rsim = diff(log(Paths),1,1);
    SimRets{i} = Rsim;
    p = 1;
    r_ts = Rsim(:,p);
    r_ts = r_ts(~isnan(r_ts));
    SharpeSim(i) = (mean(r_ts)*252-rf)/(std(r_ts)*sqrt(252));
    muAdj = mu+0.15*betaVec(i)*MarketRegime;
    signal = max(min(muSignal/(sig+eps),2),-2);
    position = 0.5+0.25*signal;
    r_ts = position*r_ts;
    % muAnn = mean(Rsim(:),'omitnan')*252;
    % volAnn = std(Rsim(:),'omitnan')*sqrt(252);
    pathMu = mean(Rsim,1)*252;
    pathSig = std(Rsim,0,1)*sqrt(252);
    
    SimMeanReturn(i) = mean(pathMu);
    SimVol(i) = mean(pathSig);
    % SharpeSim(i) = (mean(Rsim(:))*252-rf)/(std(Rsim(:))*sqrt(252));
    pathRet = mean(Rsim,1);
    % SharpeSim(i) = (mean(pathRet)*252-rf)/(std(pathRet)*sqrt(252));
    cumLogRet = sum(Rsim,1);
    annRet = exp(cumLogRet/nYears)-1;
    SharpeSim(i) = (mean(annRet)-rf)/std(annRet);
    % Quote target probability
    % targetStep  = round(targetHorizonYears/nYears*steps);
    % S0          = price(find(~isnan(price),1,'last'));
    targetPrice = targetMultiplier*S0;
    RegY{i} = y;
    RegYHat{i} = yHat;
    SimEndPaths{i} = Paths(end,:);
    TargetPriceVec(i) = targetPrice;

    % Values for Plot

    TargetProb(i)       = mean(Paths(targetStep,:)>=targetPrice);
    TargetProbHit(i)    = mean(any(Paths>=targetPrice,1));
    
    hitT = nan(1,nPaths);
    for p = 1:nPaths
        h = find(Paths(:,p)>=targetPrice,1);
        if ~isempty(h)
            hitT(p)=h*dt;
        end
    end
    TargetHitTime(i) = mean(hitT,'omitnan');
    end

    %%
    % Output Stats
    % SimMeanPrice(i) = mean(Paths(end,:));
    % SimVol(i)       = std(Paths(end,:));
    % simRet = diff(log(Paths),1,1);
    % simRet = simRet(:);
    % if std(simRet) > 0
    % SharpeSim(i) = (mean(simRet)*252-rf)/(std(simRet)*sqrt(252));
    % else
    % SharpeSim(i) = NaN;
    % end
    valid = ~cellfun(@isempty,SimRets);

    Rmat = cell2mat(cellfun(@(x) mean(x,2), SimRets(valid), 'UniformOutput',false));

    T  = min(cellfun(@(x) size(x,1),SimRets(valid)));
    Rmat = zeros(T,sum(valid));
    k = 1;
    for j = find(valid)'
        Rmat(:,k) = SimRets{j}(1:T,1);
        k = k+1;
    end
    muP = mean(Rmat)*252;
    SigmaP = cov(Rmat)*252;
    w = SigmaP\muP';
    w = w/sum(abs(w));
    portRet = Rmat*w;
    SharpePortfolio = (mean(portRet)*252-rf)/(std(portRet)*sqrt(252));
    
    % [w,SharpePortfolio] = build_max_sharpe_portfolio(Rmat,rf);
    fprintf('\nPortfolio Sharpe Ratio: %.2f\n',SharpePortfolio);

    % Paths = simulate_GBM_Merton( ...
      %   muGBM, ...
      %  sigmaGBM, ...
      %   params.lambdaJ, ...
      %   params.muJ, ...
      %   params.sigJ, ...
      %   S0,nYears,dt,nPaths);
%%
% Table size check
disp('--- Table size check ---')
vars = {
    'assetNames',assetNames(:)
    'MeanReturn',SimMeanReturn'
    'StdReturn',SimVol
    'Sharpe',SharpeSim
    'TargetProb',TargetProb
    'TargetProbHit',TargetProbHit
    'Beta',betaVec
    'TargetHitTime',TargetHitTime
    'RegSlope',RegSlope
    'RegIntercept',RegIntercept
    'RegR2',RegR2
    'RegTargetPrice',RegTargetPrice
    'RegY',RegY
    'RegYHat',RegYHat
    }

%%

vars = { ...
    assetNames(:), betaVec, SimMeanReturn, SimVol, SharpeSim, ...
    TargetProb, TargetProbHit, TargetHitTime, ...
    RegSlope, RegIntercept, RegR2, RegTargetPrice };

for k = 1:numel(vars)
    disp(size(vars{k}))
end

% assert(numel(TargetProbHit) == numel(validAsset), ...
%     'Length mismatch: TargetProbHit vs validAsset');;

% Reason = strings(nAssets,1);
% for i = 1:nAssets
%     if numel(PF(:,i)) < regWindow || all(isnan(PF(:,i)))
% Reason(i) = "Insufficient History";
%     elseif isnan(RegSlope(i))
% Reason(i) = "Regression failed";
%     elseif isnan(TargetProbHit(i))
% Reason(i)  = "Target never reached";
%     else
% Reason(i) = "Ok";
%     end 
% end

% Summary.Reason = Reason;

%%
% Summary Table
Summary = table(assetNames(:), ...
    betaVec, ...
    SimMeanReturn', ...
    SimVol, ...
    SharpeSim, ...
    TargetProb, ...
    TargetProbHit, ...
    TargetHitTime, ...
    RegSlope, ...
    RegIntercept, ...
    RegR2, ...
    RegTargetPrice, ...
    'VariableNames',{'Asset','Beta','MeanReturn','Vol','Sharpe','Prob_TargetAtHorizon','Prob_EverHit','ExpectedHitTime', ...
    'RegSlope','RegIntercept','RegR2','Reg_TargetPrice'});

disp(Summary);
fprintf('\nPortfolio Sharpe; %.2f\n',SharpePortfolio);
%%
f2 = figure('Name','Quota Target','Position',[200 200 1000 400]);
uitable('Parent',f2, ...
    'Data',table2cell(Summary), ...
    'ColumnName',Summary.Properties.VariableNames, ...
    'RowName',[], ...
    'Units','normalized', ...
    'Position',[0 0 1 1]);


%%
% Plot deception
for i = 1:nAssets
    if isempty(RegY{i})
        continue
    end
   
figure('Name',['Regression -', assetNames{i}],'Visible','on');
plot(RegY{i},'k','LineWidth',1.2);
hold on;
plot(RegYHat{i},'r--','LineWidth',1.5);
grid on;
title(sprintf('%s, | R^2 = %.2f',assetNames{i},RegR2(i)));
xlabel('Time (last 35 Obs.)');
ylabel('log(Price)');
legend('Observed','Regression','Location','best');
drawnow;
end

assert(any(~cellfun(@isempty,RegY)),'No regression data stored');

for i = 1:nAssets
    if isempty(SimEndPaths{i})
        continue
    end

figure('Name',['Target Distribution -',assetNames{i}],'Visible','on');
histogram(SimEndPaths{i},50,'Normalization','pdf');
hold on;
xline(TargetPriceVec(i),'r','LineWidth',2);
title(sprintf('%s | Target = %.2f',assetNames{i},TargetPriceVec(i)));
xlabel('Terminal Price');
ylabel('Density');
grid on
drawnow;
end
%%
whos RegY RegYHat SimEndPaths
cellfun(@isempty,RegY)
sum(~cellfun(@isempty,RegY))
%%
fprintf('Stored regression assets: %d / %d\n', ...
    sum(~cellfun(@isempty,RegY)),nAssets);
%% Function section

function params = estimate_OU_jump(logP,dt)
ret = diff(logP);
x = logP(1:end-1);
y = logP(2:end);
B = [ones(length(x),1),x]\y;
phi = min(max(B(2),0.001),0.999);
params.kappa = -log(phi)/dt;
params.theta = B(1)/(1-phi);

eps = y-(B(1)+B(2)*x);
params.sigma = std(eps)/sqrt(dt);
mu_d = mean(ret);
sig_d = std(ret);
jumpIdx = abs(ret-mu_d)>3*sig_d;
params.lambdaJ = min(mean(jumpIdx)/dt,0.2);

if any(jumpIdx)
    params.muJ = mean(ret(jumpIdx));
    params.sigJ = std(ret(jumpIdx));
else
    params.muJ = 0;
    params.sigJ = 0;
end
end
%%
function Paths = simulate_OU_Merton(params,S0,T,dt,nPaths)
N = round(T/dt);
logS = zeros(N,nPaths);
logs(1,:)=log(S0);

for t=1:N-1
    dW = randn(1,nPaths);
    J = (rand(1,nPaths)<params.lambdaJ*dt) ...
        .* (params.muJ+params.sigJ*randn(1,nPaths));
    % thetaAdj = params.theta+lambdaReg*Regime;
    logS(t+1,:) = logS(t,:) ...
        +params.kappa*(params.theta-logS(t,:))*dt ...
        +params.sigma*sqrt(dt).*dW+J;
end
Paths = exp(logS);
end

% GBM-Merton SDE
function Paths = simulate_GBM_Merton(mu,sigma,lambdaJ,muJ,sigJ,S0,T,dt,nPaths)
N = round(T/dt);
logS = zeros(N,nPaths);
logS(1,:) = log(S0);
for t=1:N-1
    dW = randn(1,nPaths);
    J = (rand(1,nPaths)<lambdaJ*dt) ...
        .* (muJ+sigJ*randn(1,nPaths));
    logS(t+1,:) = logS(t,:) ...
        +(mu-0.5*sigma^2)*dt ...
        +sigma*sqrt(dt).*dW+J;
end
Paths = exp(logS);
end

% max sharpe portfolio builder
function [w,sharpe] = build_max_sharpe_portfolio(Rsim,rf)
mu = mean(Rsim,'omitnan');
Sigma = cov(Rsim,'partialrows');

w = Sigma\mu';
w = w/sum(abs(w));
portRet = Rsim*w;
sharpe = (mean(portRet)*252-rf)/(std(portRet)*sqrt(252));
end

function params = estimate_jump_params(logP,dt)
ret = diff(logP);
mu = mean(ret);
sig = std(ret);
jumpIdx = abs(ret-mu) > 2*sig;
params.lambdaJ = min(mean(jumpIdx)/dt,0.2);

if any(jumpIdx)
    params.muJ = mean(ret(jumpIdx));
    params.sigJ = std(ret(jumpIdx));
else
    params.muJ = 0;
    params.sigJ = 0;
end
end