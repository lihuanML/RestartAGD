function [OBJ1,GRAD1,OBJ2,GRAD2,time,reiter1,reiter2] = emp_ragd(data_train,I_train,J_train,siz,N,eta)

tic 

m = siz.m; n = siz.n;  r = siz.r; 
len = length(I_train);                                   % sample size
Mat = sparse(I_train,J_train,data_train,m,n);            % a sparse matrix used to compute the gradient efficiently  

X = sparse(I_train,J_train,data_train,m,n);
[A,Sigma,B] = svds(X,r);
Sigma = sqrt(Sigma);
Ucur = A*Sigma;                           % initializaton using the SVD of the observed sparse matrix
Vcur = B*Sigma;                           % U; m*r; V: n*r

time = zeros(N,1);                        % running time
GRAD1 = zeros(N,1);                       % gradient norm; plot the figures using time as the x axis
OBJ1 = zeros(N,1);                        % objective function value; plot the figures using time as the x axis
GRAD2 = zeros(N*2,1);                     % gradient norm; plot the figures using the number of gradient and function evaluations as the x axis
OBJ2 = zeros(N*2,1);                      % objective function value; plot the figures using the number of gradient and function evaluations as the x axis
reiter1 = [];                             % record where restart occurs
reiter2 = [];                             % record where restart occurs
   
rt = 1;                                   % iteration number during each restart period
t = 0;                                    % number of restarts
itercount = 0;                            % number of gradient and function evaluations used to plot the figures

Upre = Ucur;                              % Ucur, Vcur: values at the current iteration;
Vpre = Vcur;                              % Upre, Vpre: values at the previous iteration;

preobj = funvalue_1mc(data_train,I_train,J_train,len,Ucur,Vcur);       % compute the function value

for k = 1:N                               % run N iterations
    
    %%%%%%%%%%%%%%%%%% AGD step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    beta = (rt-1)/(rt+2);
    
    TUcur = Ucur+beta*(Ucur-Upre);
    TVcur = Vcur+beta*(Vcur-Vpre);
    
    [gradU,gradV] = grad_1mc(data_train,I_train,J_train,Mat,len,TUcur,TVcur);
    % calculate the gradient at (TUcur,TVcur)

    Upre = Ucur;
    Vpre = Vcur;

    Ucur = TUcur-eta*gradU;                                              % gradient descent step at (TUcur,TVcur)
    Vcur = TVcur-eta*gradV;

    %%%%%%%%%%%%%%%%%%%% check the restart condition %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    curobj = funvalue_1mc(data_train,I_train,J_train,len,Ucur,Vcur);     % compute the function value at (Ucur,Vcur)
    % note that empirical AGD needs to calculate the gradient at (TUcur,TVcur) and the function value at (Ucur,Vcur). 
    % Both are used in the implementation of the algorithm, rather than only plotting the figures. 
    
    if curobj <= preobj                                                            % objective value decreases, go on AGD update;
        rt = rt+1;
    else                                                                           % objective increases, restart
        rt = 1;                                                                    % then beta=(rt-1)/(rt+2)=0
        t = t+1;
        reiter1(t) = k;                                                            % record where restart occurs
        reiter2(t) = itercount+1;
        disp(['restart  ',num2str(k)]);
    end
    preobj = curobj;
    
    %%%%%%%%%%%%%%%%%% record the information used to plot the figures  %%%%%%%%%%%%%%%%%%
   
    timecount = toc;
    
    GRAD1(k) = sqrt( norm(gradU,'fro')^2 + norm(gradV,'fro')^2 );                  % gradient norm at the kth iteration 
    OBJ1(k) = curobj;                                                              % objective function value at the kth iteration
    itercount = itercount + 1;                                                     % one gradient calculation at (TUcur,TVcur)
    OBJ2(itercount) = OBJ1(k);
    GRAD2(itercount) = GRAD1(k);
    itercount = itercount + 1;                                                     % one function value calculation at (Ucur,Vcur)
    OBJ2(itercount) = OBJ1(k);
    GRAD2(itercount) = GRAD1(k);
    
    if k>1
        time(k) = time(k-1)+timecount;
    else
        time(k) = timecount;
    end                                                                            % running time before the kth iteration
          
    if mod(k,100) == 0                                                             % print every 100 iterations
        disp([ 'ERAGD--iter: ',num2str(k),', gradient: ',num2str(GRAD1(k),15),', OBJ1: ',num2str(OBJ1(k),15) ] );
    end
    tic
end 