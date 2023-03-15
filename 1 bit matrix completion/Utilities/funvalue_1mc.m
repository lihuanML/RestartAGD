function [obj] = funvalue_1mc(data_train,I_train,J_train,len,U,V)

M = partXY(U',V',I_train,J_train,len);            % calculate Proj(U*V') and store it in the vector M, where Proj means the projection onto the index set of the observed samples

% denote f(x)=e^x/(1+e^x)
% denote F2(x)=-F1(x)=log f(x) if y=1 and F2(x)=-F1(x)=log(1-f(x)) if y=-1
f = 1 ./ (1 + exp(-M));                                                       % f(x)=e^x/(1+e^x)=1/(1+e^{-x})
ttlog = log( data_train.*f' - (data_train-1)/2 );                             % F2(x)=log f(x) if y=1 and F2(x)=log(1-f(x)) if y=-1
ttlog(ttlog==-inf & data_train==1) = M(ttlog==-inf & data_train==1);          % when x approx -inf, log f(x) approx -inf. We let log f(x) approx x to avoid inf because log(exp(-1000))=-inf in matlab
ttlog(ttlog==-inf & data_train==-1) = -M(ttlog==-inf & data_train==-1);       % when x approx inf, log(1-f(x)) approx -inf. We let log(1-f(x)) approx -x to avoid inf 

obj = ( -sum( ttlog ) + norm((U'*U - V'*V),'fro')^2/2 )/len;