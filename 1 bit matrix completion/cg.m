function [OBJ1,GRAD1,OBJ2,GRAD2,time] = cg(data_train,I_train,J_train,siz,N,eta)

tic

m = siz.m; n = siz.n;  r = siz.r; 
len = length(I_train);                                   % sample size
Mat = sparse(I_train,J_train,data_train,m,n);            % a sparse matrix used to compute the gradient efficiently  

X = sparse(I_train,J_train,data_train,m,n);
[A,Sigma,B] = svds(X,r);
Sigma = sqrt(Sigma);
U = A*Sigma;                                             % initializaton using the SVD of the observed sparse matrix
V = B*Sigma;                                             % U; m*r; V: n*r
  
time = zeros(N,1);                                       % running time
GRAD1 = zeros(N,1);                                      % gradient norm; plot the figures using time as the x axis   
OBJ1 = zeros(N,1);                                       % objective function value; plot the figures using time as the x axis
GRAD2 = zeros(N*3,1);                                    % gradient norm; plot the figures using the number of gradient and function evaluations as the x axis
OBJ2 = zeros(N*3,1);                                     % objective function value; plot the figures using the number of gradient and function evaluations as the x axis

[gradU,gradV,curobj] = grad_funvalue_1mc(data_train,I_train,J_train,Mat,len,U,V);
% compute the gradient and function value at (U,V)
    
gradU_pre = gradU;                                       % gradient at previous iteration
gradV_pre = gradV;
direction_U = 0;                                         % update direction
direction_V = 0;

t = 0;
itercount = 0;                                           % number of gradient and function evaluations used to plot the figures.

for k = 1:N                                              % run N iterations
     
    %%%%%%%%%%%%%%%%%%%% calculate the update direction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    beta = ( sum(sum( gradU.*(gradU-gradU_pre) )) + sum(sum( gradV.*(gradV-gradV_pre) )) )/( norm(gradU_pre,'fro')^2 + norm(gradV_pre,'fro')^2 );
    beta = max([beta,0]);
    direction_U = - gradU + beta * direction_U;
    direction_V = - gradV + beta * direction_V;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% backtracking line search %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    teta = eta*2;
    
    while 1
        t = t+1;
        
        tU = U + teta * direction_U; 
        tV = V + teta * direction_V;                                                      % update
   
        M = partXY(tU',tV',I_train,J_train,len);     
        % denote f(x)=e^x/(1+e^x)
        % denote F2(x)=log f(x) if y=1 and F2(x)=log(1-f(x)) if y=-1
        f = 1 ./ (1 + exp(-M));                                                           % f(x)=e^x/(1+e^x)=1/(1+e^{-x})
        ttlog = log( data_train.*f' - (data_train-1)/2 );                                 % F2(x)=log f(x) if y=1 and F2(x)=log(1-f(x)) if y=-1
        ttlog(ttlog==-inf & data_train==1) = M(ttlog==-inf & data_train==1);              % when x approx -inf, log f(x) approx -inf. We let log f(x) approx x to avoid inf because log(exp(-1000))=-inf in matlab
        ttlog(ttlog==-inf & data_train==-1) = -M(ttlog==-inf & data_train==-1);           % when x approx inf, log(1-f(x)) approx -inf. We let log(1-f(x)) approx -x to avoid inf 
        nextobj = ( -sum(ttlog) + norm((tU'*tU - tV'*tV),'fro')^2/2 )/len;                % compute the function value at (tU,tV)
                   
        itercount = itercount + 1;                                                        %  one more functino value calculation at (tU,tV)
        OBJ2(itercount) = curobj;
        GRAD2(itercount) = sqrt( norm(gradU,'fro')^2 + norm(gradV,'fro')^2 );

        if nextobj <= curobj + teta * ( sum(sum(direction_U.*gradU)) + sum(sum(direction_V.*gradV)) )/2 || t>10
            % at most 10 line searches
            t = 0;
            
            eta = teta;                                                                   % keep the new step size
            U = tU;
            V = tV;
            gradU_pre = gradU;
            gradV_pre = gradV;                                                            % store the previous gradient
            
            % denote f(x)=e^x/(1+e^x)
            % denote F1(x)=-log f(x)=-x+log(1+e^x) if y=1 and F1(x)=-log(1-f(x))=log(1+e^x) if y=-1
            % then nabla F1(x)=-1+e^x/(1+e^x)=-1/(1+e^x) if y=1 and nabla F1(x)=e^x/(1+e^x) if y=-1
            G = -1./(1+exp(M')) - (data_train-1)/2;                                       % G is nabla F1(x), M is x, and data_train is y (1 or -1)
            updateSval1(Mat, G', len);                                                    % recover the sparse matrix Mat from the vector G
                                                                                          % That is, Mat = sparse(I_train,J_train,G,m,n). But the matlab code is much slower than the C code. 
            gradU = ( Mat * V + U*(U'*U - V'*V) )/len;
            gradV = ( Mat' * U + V*(V'*V - U'*U) )/len;                                   % compute the current gradient at (U,V)
            curobj = nextobj;
            break;                                                                        % break line search
        else
            teta = teta/2;                                                                % decrease the step size and go on line search
        end
    end
    
    %%%%%%%%%%%%%%%%%% record the information used to plot the figures  %%%%%%%%%%%%%%%%%%
    
    
    timecount = toc;
    
    GRAD1(k) = sqrt( norm(gradU,'fro')^2 + norm(gradV,'fro')^2 );                         % gradient norm at the kth iteration 
    OBJ1(k) = curobj;                                                                     % objective function value at the kth iteration
    itercount = itercount + 1;                                                            % one gradient calculation at (U,V) in the if branch
    OBJ2(itercount) = curobj;
    GRAD2(itercount) = GRAD1(k);
      
    if k>1
        time(k) = time(k-1)+timecount;
    else
        time(k) = timecount;
    end                                                                                   % running time before the kth iteration
          
    if mod(k,100) == 0                                                                    % print every 100 iterations
        disp([ 'CG--iter: ',num2str(k),', gradient: ',num2str(GRAD1(k),15),', obj: ',num2str(OBJ1(k),15) ] );
    end
    tic
end
OBJ2 = OBJ2(1:itercount);
GRAD2 = GRAD2(1:itercount);