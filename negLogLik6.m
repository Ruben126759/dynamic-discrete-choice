%% negLogLik.m
% The function computes minus the log partial likelihood and, optionally, the corresponding minus score and information matrix estimate for conditional choice data on the basic firm entry and exit model used as an example in CentER's Empirical Industrial Organization II.

%{
The function |negLogLik| computes minus the log partial likelihood for the conditional choice part of the data. Optionally, it also returns minus the corresponding score vector and an estimate of the information matrix for the parameter (sub)vector $\theta\equiv(\beta_0,\beta_1,\delta_1)'$ (the scores are specific to the estimation example in Section \ref{script}'s script and should be adapted for inference on other parameters).
%}
function nll = ...
         negLogLik6(choices,iX,supportX,capPi,beta,delta1,delta2,Proptype1,rho,flowpayoffs,bellman,fixedPoint,tolFixedPoint)
%{
The function |negLogLik| requires the following input arguments:
	\begin{dictionary}
	\item{|choices|} a $T\times N$ matrix with choice observations $a_{tn}$;
	\item{|iX|} a $T\times N$ matrix with indices of observed states $x_{tn}$ in ${\cal X}$ (for example, if $x_{11}=x^3$, then the first element of |iX| is 3, not $x^3$); 
	\item{|supportX|} a $K\times 1$ vector with the support points of the profit state $X_t$ (the elements of $\cal{X}$, consistently ordered with the Markov transition matrix $\Pi$);
	\item{|capPi|} the (possibly estimated) $K\times K$ Markov transition matrix $\Pi$ for $\{X_t\}$, with typical element $\Pi_{ij}=\Pr(X_{t+1}=x^j|X_t=x^i)$;
	\item{|beta|} a $2\times 1$ vector that contains the intercept ($\beta_0$) and profit state slope ($\beta_1$) of the flow payoffs to choice $1$;
	\item{|delta|} a $2\times 1$ vector that contains the firm's exit ($\delta_0$) and entry ($\delta_1$) costs;	
	\item{|rho|} a scalar with the value of the discount factor $\rho$;
	\item{|flowpayoffs|} a handle of a function |[u0,u1]=flowpayoffs(supportX,beta,delta)| that computes the mean flow payoffs $u_0$ and $u_1$;
	\item{|bellman|} a handle of a function |[capU0,capU1] = bellman(capU0,capU1,u0,u1,capPi,rho)| that iterates once on $\Psi$;
	\item{|fixedPoint|} a handle of a function |[capU0,capU1] = fixedPoint(u0,u1,capPi,rho,tolFixedPoint,bellman,capU0,capU1)| that computes the fixed point $U$ of $\Psi$; and
	\item{|tolFixedPoint|} a scalar tolerance level that is used to determine convergence of the successive approximations of the fixed point $U$ of $\Psi$.
	\end{dictionary}
	It returns
	\begin{dictionary}
	\item{|nll|} a scalar with minus the log partial likelihood for the conditional choices
	\end{dictionary}
    and optionally
	\begin{dictionary}
	\item{|negScore|} a $3\times 1$ vector with minus the partial likelihood score for $\theta$ and
    \item{|informationMatrix|} a $3\times 3$ matrix with the sum of the $N$ outer products of the individual contributions to the score for $\theta$.
	\end{dictionary}
The function |negLogLik| first stores the number $K$ of elements of |supportX| in a scalar |nSuppX|.
%}
nSuppX = size(supportX,1);
%{
Next, it computes the flow payoffs $u_0$ (|u0|) and $u_1$ (|u1|), the choice-specific net expected discounted values $U_0$ (|capU0|) and $U_1$ (|capU1|), their contrast $\Delta U$ (|deltaU|), and the implied probabilities $1/\left[1+\exp(\Delta U)\right]$ of not serving the market (|pExit|) for the inputted parameter values. Note that this implements the NFXP procedure's inner loop.
%}
[u0type1,u1type1] = flowpayoffs(supportX,beta,delta1); 
[capU0type1,capU1type1] = fixedPoint(u0type1,u1type1,capPi,rho,tolFixedPoint,bellman,[],[]);
deltaUtype1 = capU1type1-capU0type1;

[u0type2,u1type2] = flowpayoffs(supportX,beta,delta2); 
[capU0type2,capU1type2] = fixedPoint(u0type2,u1type2,capPi,rho,tolFixedPoint,bellman,[],[]);
deltaUtype2 = capU1type2-capU0type2;

pExittype1 = 1./(1+exp(deltaUtype1));
pExittype2 = 1./(1+exp(deltaUtype2));
%{
\paragraph{Log Partial Likelihood}
The contribution to the likelihood of firm $n$'s choice in period $t$ is the conditional choice probability 
	\[p(a_{tn}|x_{tn},a_{(t-1)n})=a_{tn}+\frac{1-2 a_{tn} }{1+\exp\left[\Delta U(x_{tn},a_{(t-1)n})\right]},\] 
with $a_{0n}=0$. The function |negLogLik| first computes these probabilities for each firm $n$ and period $t$ and stores them in a $T\times N$ matrix |p|. Then, it returns minus the sum of their logs, the log partial likelihood for the conditional choices, in |nll|. 
%}
laggedChoices = [zeros(1,size(choices,2));choices(1:end-1,:)];
ptype1 = choices + (1-2*choices).*pExittype1(iX+nSuppX*laggedChoices);
ptype2 = choices + (1-2*choices).*pExittype2(iX+nSuppX*laggedChoices);
nll = -sum(log(Proptype1*prod(ptype1)+(1-Proptype1)*prod(ptype2)));