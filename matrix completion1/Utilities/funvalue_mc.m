function [obj] = funvalue_mc(data_train,I_train,J_train,len,U,V)

M = partXY(U',V',I_train,J_train,len);               % calculate Proj(U*V') and store it in the vector M, where Proj means the projection onto the index set of the observed samples
mat = M' - data_train;                               % calculate Proj(U*V'-X*) and store it in the vector mat; X* is the observed sample matrix

obj = ( norm(mat)^2 + norm((U'*U - V'*V),'fro')^2 )/2/len;         % calculate the function value at (U,V)