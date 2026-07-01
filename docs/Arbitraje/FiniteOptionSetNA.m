function ToF=FiniteOptionSetNA(factor,forward,Strikes,CallPrices)
% Checks arbitrage in a finite set of options.

% Variables 
K=sortrows(Strikes(:));
C=-sortrows(-CallPrices(:))';
n=length(K);
L=[0 K'];
K=repmat(K,1,n+1);
L=repmat(L,n,1); % aux. matrix
% V vector
V=[1/factor,forward/factor,C];
% A matrix init.
A=zeros(n+2);
% A filling
A(2:n+1,2:n+2)=max((K-L),0);
A(1:n+1,1)=1;
A(n+2,2:n+2)=1;
% Checking product
ToF=all((V/A)>0);
