function [gradU,gradV,obj] = grad_funvalue_mc(data_train,I_train,J_train,Mat,len,U,V)

M = partXY(U',V',I_train,J_train,len);               % calculate Proj(U*V') and store it in the vector M, where Proj means the projection onto the index set of the observed samples
mat = M' - data_train;                               % calculate Proj(U*V'-X*) and store it in the vector mat; X* is the observed sample matrix

updateSval1(Mat, mat', len);                         % recover the sparse matrix Mat from the vector mat 
                                                     % That is, Mat = sparse(I_train,J_train,mat,m,n). But the matlab code is much slower than the C code.
                                                     % Mat = (U*V'-X*) .* supp, where supp is the support of the observed elements in X*
gradU = ( Mat * V + U*(U'*U - V'*V) )/len;
gradV = ( Mat' * U + V*(V'*V - U'*U) )/len;          % calculate the gradient at (U,V)

obj = ( norm(mat)^2 + norm((U'*U - V'*V),'fro')^2 )/2/len;         % calculate the function value at (U,V)