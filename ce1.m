%{
\subsection{Simulating Data}
	
	First, we set the number of time periods (|nPeriods|) and firms (|nFirms|) that we would like to have in our sample.
%}

clc
clear

%nSim = 10;
%nSim = 100;
nSim = 1000;
Estimates1=nan(nSim, 3);
StdErr1=nan(nSim, 3);
Estimates2=nan(nSim, 3);
StdErr2=nan(nSim, 3);
Estimates3=nan(nSim, 3);
StdErr3=nan(nSim, 3);

for j=1:3
nPeriods = 100;
nFirms = 10^j;
rng('default')
%{
	We also set the tolerance |tolFixedPoint| on the fixed point $U$ of $\Psi$ that we will use to determine the simulation's entry and exit rules. This same tolerance will also be used when solving the model in the inner loop of the NFXP procedure.
%}
tolFixedPoint = 1e-10;
%{
Next, we specify the values of the model's parameters used in the simulation: 
	\begin{dictionary}
	\item{|nSuppX|} the scalar number $K$ of elements of ${\cal X}$;
	\item{|supportX|} the $K\times 1$ vector ${\cal X}$ with the support points of $X_t$;	
	\item{|capPi|} the $K\times K$ Markov transition matrix $\Pi$ for $\{X_t\}$, with typical element $\Pi_{ij}=\Pr(X_{t+1}=x^j|X_t=x^i)$;
	\item{|beta|} the $2\times 1$ vector $\beta$ with the parameters of the flow profit of active firms;
	\item{|delta|} the $2\times 1$ vector of exit and entry costs $\delta$; and
	\item{|rho|} the scalar discount factor $\rho$.
	\end{dictionary}
	%}													
nSuppX = 5;
supportX = (1:nSuppX)';
capPi = 1./(1+abs(ones(nSuppX,1)*(1:nSuppX)-(1:nSuppX)'*ones(1,nSuppX)));
capPi = capPi./(sum(capPi')'*ones(1,nSuppX));
beta = [-0.1*nSuppX;0.2];
delta = [0;1];
rho = 0.95	;
%{
For these parameter values, we compute the flow payoffs $u_0$ (|u0|) and $u_1$ (|u1|), the choice-specific expected discounted values $U_0$ (|capU0|) and $U_1$ (|capU1|), and their contrast $\Delta U$ (|deltaU|).
%}
[u0,u1] = flowpayoffs(supportX,beta,delta); 
[capU0,capU1] = fixedPoint(u0,u1,capPi,rho,tolFixedPoint,@bellman,[],[]);
deltaU = capU1-capU0;
%{
	With $\Delta U$ computed, and $\Pi$ specified, we proceed to simulate a $T\times N$ matrix of choices |choices| and a $T\times N$ matrix of states |iX| (recall from Section \ref{simulate} that |iX| contains indices that point to elements of ${\cal X}$ rather than those values themselves).
%}

for i=1:nSim
[choices,iX] = simulateData(deltaU,capPi,nPeriods,nFirms);
%{
\subsection{Nested Fixed Point Maximum Likelihood Estimation}

First, suppose that $\Pi$ is known. We use |fmincon| from \textsc{Matlab}'s \textsc{Optimization Toolbox} to maximize the partial likelihood for the choices (the code can easily be adapted to use other optimizers and packages, because these have a very similar \url{http://www.mathworks.nl/help/optim/ug/fmincon.html}{syntax}; see below). Because |fmincon| is a minimizer, we use minus the log likelihood as its objective. The function |negLogLik| computes this objective, but has input arguments other than the vector of model parameters to be estimated. Because \url{http://www.mathworks.nl/help/optim/ug/passing-extra-parameters.html}{the syntax of |fmincon| does not allow this}, we define a function handle |objectiveFunction| to an anonymous function that equals |negLogLik| but does not have this extra inputs.
%}
objectiveFunction = @(parameters)negLogLik(choices,iX,supportX,capPi,parameters(1:2),[delta(1);parameters(3)],...
                                           rho,@flowpayoffs,@bellman,@fixedPoint,tolFixedPoint);
%{
Before we can put |fmincon| to work on this objective function, we first have to set some of its other input arguments. We specify a $3\times 1$ vector |startvalues| with starting values for the parameters to be estimated, $(\beta_0,\beta_1,\delta_1)'$.
%}
startvalues = [randn(2,1); abs(randn(1))];
%{
    We also set a lower bound of 0 on the third parameter, $\delta_1$, and (nonbinding) lower bounds of $-\infty$ on the other two parameters (|lowerBounds|). There is no need to specify upper bounds.\footnote{Note that |fmincon|, but also its alternatives discussed below, allow the user to specify bounds on parameters; if another function is used that does not allow for bounds on the parameters, you can use an alternative parameterization to ensure that parameters only take values in some admissible set (for example, you can specify $\delta_1=\exp(\delta_1^*)$ for $\delta_1^*\in\mathbb{R}$ to ensure that $\delta_1>0$). Minimizers like |fmincon| also allow you to impose more elaborate constraints on the parameters; you will need this option when implementing the MPEC alternative to NFXP of \cite{ecta12:juddsu} (see Section \ref{exercises}).}
%}
lowerBounds = -Inf*ones(size(startvalues));
lowerBounds(3) = 0;
%{
    Finally, we pass some options, including tolerances that specify the criterion for the outer loop convergence, to |fmincon| through the structure |OptimizerOptions| (recall that we have already set the inner loop tolerance in |tolFixedPoint|). We use the function |optimset| from the \textsc{Optimization Toolbox} to assign values to specific fields (options) in |OptimizerOptions| and then call |fmincon| to run the NFXP maximum likelihood procedure (to use \textsc{Knitro} instead, simply replace |fmincon| by |knitromatlab|, |knitrolink|, or |ktrlink|, depending on the packages installed\footnote{|fmincon| requires \textsc{Matlab}'s \textsc{Optimization Toolbox}, |knitromatlab| is included in \textsc{Knitro} 9.0, |knitrolink| uses both, and |ktrlink| can be used if the \textsc{Optimization Toolbox} is installed with an earlier version of \textsc{Knitro}.}).
%}
OptimizerOptions = optimset('Display','off','Algorithm','interior-point','AlwaysHonorConstraints','bounds',...
                            'GradObj','on','TolFun',1E-6,'TolX',1E-10,'DerivativeCheck','off','TypicalX',[beta;delta(2)]);
[maxLikEstimates,~,exitflag] = fmincon(objectiveFunction,startvalues,[],[],[],[],lowerBounds,[],[],OptimizerOptions);
%{
This gives maximum partial likelihood estimates of $(\beta_0,\beta_1,\delta_1)$. To calculate standard errors, we call |negLogLik| once more to estimate the corresponding Fisher information matrix and store this in |informationMatrix|. Its inverse is an estimate of the maximum likelihood estimator's asymptotic variance-covariance matrix.
%}
[~,~,informationMatrix] = objectiveFunction(maxLikEstimates);
standardErrors = diag(sqrt(inv(informationMatrix)));
if j==1
Estimates1(i,:)=maxLikEstimates;
StdErr1(i,:)=standardErrors;
elseif j==2
Estimates2(i,:)=maxLikEstimates;
StdErr2(i,:)=standardErrors;
else
Estimates3(i,:)=maxLikEstimates;
StdErr3(i,:)=standardErrors;
end
end
end
%{
The resulting parameter estimates, and numerical and analytical standard errors are displayed (column 3-5), together with the parameters' true (first column) and starting values (second column).
Avg of the esimated parameters is very close to the truth with small SE,
analytical somewhat smaller than numerical but very close.
%}
disp('Summary of Results');
disp('--------------------------------------------');
disp('     true    mean_est   ste_num   ste_an   mean_est   ste_num   ste_an   mean_est   ste_num   ste_an');
disp([[beta;delta(2)] mean(Estimates1, 1)' std(Estimates1, 1)' mean(StdErr1,1)' mean(Estimates2, 1)' std(Estimates2, 1)' mean(StdErr2,1)' mean(Estimates3, 1)' std(Estimates3, 1)' mean(StdErr3,1)']);