function [OBJ1,GRAD1,OBJ2,GRAD2,time] = cg(data_train,I_train,J_train,siz,N,eta)

tic

m = siz.m; n = siz.n;  r = siz.r; 
len = length(I_train);                                     % sample size
Mat = sparse(I_train,J_train,data_train,m,n);              % a sparse matrix used to compute the gradient efficiently  

X = sparse(I_train,J_train,data_train,m,n);
[A,Sigma,B] = svds(X,r);
Sigma = sqrt(Sigma);
U = A*Sigma;                         % initializaton using the SVD of the observed sparse matrix
V = B*Sigma;                         % U; m*r; V: n*r

time = zeros(N,1);                   % running time
GRAD1 = zeros(N,1);                  % gradient norm; plot the figures using time as the x axis   
OBJ1 = zeros(N,1);                   % objective function value; plot the figures using time as the x axis
GRAD2 = zeros(N*3,1);                % gradient norm; plot the figures using the number of gradient and function evaluations as the x axis
OBJ2 = zeros(N*3,1);                 % objective function value; plot the figures using the number of gradient and function evaluations as the x axis

[gradU,gradV,curobj] = grad_funvalue_mc(data_train,I_train,J_train,Mat,len,U,V);
% compute the gradient and function value at (U,V)

gradU_pre = gradU;                   % gradient at previous iteration
gradV_pre = gradV;
directionU = 0;                      % update direction
directionV = 0;                      

t = 0;
itercount = 0;                       % number of gradient and function evaluations used to plot the figures.

for k = 1:N                          % run N iterations
     
    %%%%%%%%%%%%%%%%%%%% calculate the update direction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    beta = ( sum(sum( gradU.*(gradU-gradU_pre) )) + sum(sum( gradV.*(gradV-gradV_pre) )) )/( norm(gradU_pre,'fro')^2 + norm(gradV_pre,'fro')^2 );
    beta = max([beta,0]);
    directionU = - gradU + beta * directionU;
    directionV = - gradV + beta * directionV;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% backtracking line search %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    teta = eta*2;
    
    while 1
        t = t+1;
        
        tU = U + teta * directionU; 
        tV = V + teta * directionV;                            % update
   
        M = partXY(tU',tV',I_train,J_train,len);               % calculate Proj(U*V') and store it in the vector M, where Proj means the projection onto the index set of the observed samples
        mat = M' - data_train;                                 % calculate Proj(U*V'-X*) and store it in the vector mat; X* is the observed sample matrix

        nextobj = ( norm(mat)^2 + norm((tU'*tU - tV'*tV),'fro')^2 )/2/len;         % calculate the function value at (U,V)
      
        itercount = itercount + 1;                             %  one more functino value calculation at (tU,tV)
        OBJ2(itercount) = curobj;
        GRAD2(itercount) = sqrt( norm(gradU,'fro')^2 + norm(gradV,'fro')^2 );
            
        if nextobj <= curobj + teta * ( sum(sum(directionU.*gradU)) + sum(sum(directionV.*gradV)) )/2 || t>10
            % at most 10 line searches
            t = 0;
            
            eta = teta;                                        % keep the new step size
            U = tU;
            V = tV;
            gradU_pre = gradU;
            gradV_pre = gradV;                                 % store the previous gradient
            updateSval1(Mat, mat', len);
            gradU = ( Mat * V + U*(U'*U - V'*V) )/len;
            gradV = ( Mat' * U + V*(V'*V - U'*U) )/len;        % compute the current gradient at (U,V)
            curobj = nextobj;
            break;                                             % break line search
        else
            teta = teta/2;                                     % decrease the step size and go on line search
        end
    end
    
    %%%%%%%%%%%%%%%%%% record the information used to plot the figures  %%%%%%%%%%%%%%%%%%
    
    timecount = toc;
    
    GRAD1(k) = sqrt( norm(gradU,'fro')^2 + norm(gradV,'fro')^2 );         % gradient norm at the kth iteration 
    OBJ1(k) = curobj;                                          % objective function value at the kth iteration
    itercount = itercount + 1;                                 % one gradient calculation at (U,V) in the if branch
    OBJ2(itercount) = curobj;
    GRAD2(itercount) = GRAD1(k);
      
    if k>1
        time(k) = time(k-1)+timecount;
    else
        time(k) = timecount;
    end                                                        % running time before the kth iteration
          
    if mod(k,100) == 0                                         % print every 100 iterations
        disp([ 'CG--iter: ',num2str(k),', gradient: ',num2str(GRAD1(k),15),', obj: ',num2str(OBJ1(k),15) ] );
    end
    tic
end
OBJ2 = OBJ2(1:itercount);
GRAD2 = GRAD2(1:itercount);


