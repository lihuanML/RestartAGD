function [gradU,gradV] = grad_1mc(data_train,I_train,J_train,Mat,len,U,V)

M = partXY(U',V',I_train,J_train,len);            % calculate Proj(U*V') and store it in the vector M, where Proj means the projection onto the index set of the observed samples
    
% denote f(x)=e^x/(1+e^x)
% denote F1(x)=-log f(x)=-x+log(1+e^x) if y=1 and F1(x)=-log(1-f(x))=log(1+e^x) if y=-1
% then nabla F1(x)=-1+e^x/(1+e^x)=-1/(1+e^x) if y=1 and nabla F1(x)=e^x/(1+e^x) if y=-1
G = -1./(1+exp(M')) - (data_train-1)/2;           % G is nabla F1(x), M is x, and data_train is y (1 or -1)
updateSval1(Mat, G', len);                        % recover the sparse matrix Mat from the vector G
                                                  % That is, Mat = sparse(I_train,J_train,G,m,n). But the matlab code is much slower than the C code. 
gradU = ( Mat * V + U*(U'*U - V'*V) )/len;
gradV = ( Mat' * U + V*(V'*V - U'*U) )/len;       % calculate the gradient at (U,V)
