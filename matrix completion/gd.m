function [OBJ1,GRAD1,OBJ2,GRAD2,time] = gd(data_train,I_train,J_train,siz,N,eta)

tic

m = siz.m; n = siz.n;  r = siz.r; 
len = length(I_train);                               % sample size
Mat = sparse(I_train,J_train,data_train,m,n);        % a sparse matrix used to compute the gradient efficiently  

X = sparse(I_train,J_train,data_train,m,n);
[A,Sigma,B] = svds(X,r);
Sigma = sqrt(Sigma);
U = A*Sigma;             % initializaton using the SVD of the observed sparse matrix
V = B*Sigma;             % U; m*r; V: n*r

time = zeros(N,1);       % running time
GRAD1 = zeros(N,1);      % gradient norm; plot the figures using time as the x axis
OBJ1 = zeros(N,1);       % objective function value; plot the figures using time as the x axis
GRAD2 = zeros(N,1);      % gradient norm; plot the figures using the number of gradient and function evaluations as the x axis
OBJ2 = zeros(N,1);       % objective function value; plot the figures using the number of gradient and function evaluations as the x axis

itercount = 0;           % number of gradient and function evaluations used to plot the figures.

for k = 1:N              % run N iterations
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GD step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [gradU,gradV] = grad_mc(data_train,I_train,J_train,Mat,len,U,V);
    % calculate the gradient and functino value at (U,V)
    % Note that function value is only used to plot the figures 
    
    U = U - eta * gradU;                                               % gradient descent step at (U,V)
    V = V - eta * gradV;
   
    %%%%%%%%%%%%%%%%%% record the information used to plot the figures  %%%%%%%%%%%%%%%%%%
    
    timecount = toc;
    
    OBJ1(k) = funvalue_mc(data_train,I_train,J_train,len,U,V);         % objective function value at the kth iteration
    GRAD1(k) = sqrt( norm(gradU,'fro')^2 + norm(gradV,'fro')^2 );      % gradient norm at the kth iteration 
    itercount = itercount + 1;                                         % GD computes only one gradient. Function values are only used to plot the figures
    OBJ2(itercount) = OBJ1(k);
    GRAD2(itercount) = GRAD1(k);
    
    if k>1
        time(k) = time(k-1)+timecount;
    else
        time(k) = timecount;
    end                                                  % running time before the kth iteration
          
    if mod(k,100) == 0                                   % print every 100 iterations
        disp([ 'GD--iter: ',num2str(k),', gradient: ',num2str(GRAD1(k),15),', obj: ',num2str(OBJ1(k),15) ] );
    end
    
    tic
end



