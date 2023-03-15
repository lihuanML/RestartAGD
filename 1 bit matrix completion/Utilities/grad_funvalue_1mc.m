function [gradU,gradV,obj] = grad_funvalue_1mc(data_train,I_train,J_train,Mat,len,U,V)

M = partXY(U',V',I_train,J_train,len);            % calculate Proj(U*V') and store it in the vector M, where Proj means the projection onto the index set of the observed samples
    
% denote f(x)=e^x/(1+e^x)
% denote F1(x)=-log f(x)=-x+log(1+e^x) if y=1 and F1(x)=-log(1-f(x))=log(1+e^x) if y=-1
% then nabla F1(x)=-1+e^x/(1+e^x)=-1/(1+e^x) if y=1 and nabla F1(x)=e^x/(1+e^x) if y=-1
G = -1./(1+exp(M')) - (data_train-1)/2;           % G is nabla F1(x), M is x, and data_train is y (1 or -1)
updateSval1(Mat, G', len);                        % recover the sparse matrix Mat from the vector G
                                                  % That is, Mat = sparse(I_train,J_train,G,m,n). But the matlab code is much slower than the C code. 
gradU = ( Mat * V + U*(U'*U - V'*V) )/len;
gradV = ( Mat' * U + V*(V'*V - U'*U) )/len;       % calculate the gradient at (U,V)


% also denote f(x)=e^x/(1+e^x)
% denote F2(x)=-F1(x)=log f(x) if y=1 and F2(x)=-F1(x)=log(1-f(x)) if y=-1
f = 1 ./ (1 + exp(-M));                                                       % f(x)=e^x/(1+e^x)=1/(1+e^{-x})
ttlog = log( data_train.*f' - (data_train-1)/2 );                             % F2(x)=log f(x) if y=1 and F2(x)=log(1-f(x)) if y=-1
ttlog(ttlog==-inf & data_train==1) = M(ttlog==-inf & data_train==1);          % when x approx -inf, log f(x) approx -inf. We let log f(x) approx x to avoid inf because log(exp(-1000))=-inf in matlab
ttlog(ttlog==-inf & data_train==-1) = -M(ttlog==-inf & data_train==-1);       % when x approx inf, log(1-f(x)) approx -inf. We let log(1-f(x)) approx -x to avoid inf 

obj = ( -sum( ttlog ) + norm((U'*U - V'*V),'fro')^2/2 )/len;
    