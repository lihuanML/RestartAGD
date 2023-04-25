function [OBJ1,GRAD1,OBJ2,GRAD2,time,reiter1,inciter1,biter1,reiter2,inciter2,biter2] = adp_rhb_nc(data_train,I_train,J_train,siz,N,eta)
% suggest to tune B0 and theta for different applications 
tic

m = siz.m; n = siz.n;  r = siz.r; 
len = length(I_train);                                % sample size
Mat = sparse(I_train,J_train,data_train,m,n);         % a sparse matrix used to compute the gradient efficiently  

X = sparse(I_train,J_train,data_train,m,n);
[A,Sigma,B] = svds(X,r);
Sigma = sqrt(Sigma);
Ucur = A*Sigma;             % initializaton using the SVD of the observed sparse matrix
Vcur = B*Sigma;             % U; m*r; V: n*r

time = zeros(N,1);          % running time
GRAD1 = zeros(N,1);         % gradient norm; plot the figures using time as the x axis
OBJ1 = zeros(N,1);          % objective function value; plot the figures using time as the x axis
GRAD2 = zeros(N*2,1);       % gradient norm; plot the figures using the number of gradient and function evaluations as the x axis
OBJ2 = zeros(N*2,1);        % objective function value; plot the figures using the number of gradient and function evaluations as the x axis
reiter1 = [];               % record where restart occurs
inciter1 = [];              % record where step 11 in Algorithm 4 is invoked 
reiter2 = [];               % record where restart occurs
inciter2 = [];              % record where step 11 in Algorithm 4 is invoked 
biter1 = -1;                % record where B0 decreases smaller than B for the first time
biter2 = -1;                % record where B0 decreases smaller than B for the first time

Upre = Ucur;                % Ucur, Vcur: values at the current iteration;
Vpre = Vcur;                % Upre, Vpre: values at the previous iteration;

eps = 1e-4;                 % precison. Smaller (e.g., than 1e-4) is better since theta is more likely to belong to [0,0.01]
rho = 1;                    % Hessian Lipschitz constant. Unknown and an initial guess
theta = 0.005*(eps*rho*eta^2)^(1/4);     % 1-theta is the momentum parameter
                            %%%%%%%%%%%%%% Note: theta is set smaller than the theoretical setting;
                                         % We observe that the algorithm performs better when theta is in [0,0.01]
KK = floor(1/theta);                     % parameter K in Algorithm 4
BB = sqrt(eps/rho);                      % parameter B in Algorithm 4
B0 = 100;                                % parameter B0 in Algorithm 4
                                         % can be larger to reduce the restart frequency for high dimension problems, such as DNN
[theta,BB,KK]
S = 0;                                   % cumulated norms of iterate differences. Used in step 5 in Algorithm 4
iter = 0;                                % iteratrion number
k = 0;                                   % iteratrion number during each restart period
t = 0;                                   % number of restarts
tt = 0;                                  % how many times step 11 in Algorithm 4 is invoked 
itercount = 0;                           % number of gradient and function evaluations used to plot the figures

preobj = funvalue_mc(data_train,I_train,J_train,len,Ucur,Vcur);        % compute the current function value
preUU = Ucur;                            % parameter x^0_cur in Algorithm 4
preVV = Vcur;

while k < KK || B0 > BB                  % step 2 in Algorithm 4

    %%%%%%%%%%%%%%%%%%%%%%% step 3 in Algorithm 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [gradU,gradV] = grad_mc(data_train,I_train,J_train,Mat,len,Ucur,Vcur);
    % calculate the gradient at (Ucur,Vcur)

    NUcur = Ucur-eta*gradU+(1-theta)*(Ucur-Upre);       % step 3 in Algorithm 4
    NVcur = Vcur-eta*gradV+(1-theta)*(Vcur-Vpre);

    Upre = Ucur;
    Vpre = Vcur;
    
    Ucur = NUcur;
    Vcur = NVcur;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% steps 4-13 in Algorithm 4: check the restart condition %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    k = k + 1;                                                      % step 4 in Algorithm 4
    S = S + norm(Ucur-Upre,'fro')^2 + norm(Vcur-Vpre,'fro')^2;      % cumulated norms of iterate differences
    if k*S > max([BB^2,B0^2]) || k>KK                               % step 5 in Algorithm 4
        
        ZUcur = ( Ucur + (1-2*theta)*(1-theta)*Upre ) / (1+(1-2*theta)*(1-theta));
        ZVcur = ( Vcur + (1-2*theta)*(1-theta)*Vpre ) / (1+(1-2*theta)*(1-theta));     % step 6 in Algorithm 4
           
        curobj = funvalue_mc(data_train,I_train,J_train,len,ZUcur,ZVcur);  % calculate f(z^k) used in step 8 in Algorithm 4
        
        itercount = itercount + 1;                                  % one more function value calculation at (ZUcur,ZVcur)
        OBJ2(itercount) = OBJ1(iter);
        GRAD2(itercount) = sqrt( norm(gradU,'fro')^2 + norm(gradV,'fro')^2 );
        
        t = t+1;
        B0 = B0/(1+t*0.001);                                        % step 7 in Algorithm 4 with c0=1+t*0.001
        reiter1(t) = iter;                                          % record where restart occurs
        reiter2(t) = itercount;
        if B0 <= BB && biter1 == -1                                 % record where B0 decreases smaller than B for the first time
            biter1 = iter;
            biter2 = itercount;
        end
        
        if curobj - preobj <= - eps^(1.5)/sqrt(rho) * 1e-5          % step 8 in Algorithm 4 with gamma=1e-5
                                                                    % We observe that smaller gamma makes step 11 triggered less frequently 
           disp(['Adp_RHB_NC--restart: Case 1,  [',num2str(t),' ',num2str(k),'],  iter: ',num2str(iter),',  gradient: ',num2str(GRAD1(iter),15),',  curobj: ',num2str(curobj,15),', [B,B0]: [',num2str(BB),' ',num2str(B0),']'] );
           Ucur = ZUcur;                                            % step 9 in Algorithm 4
           Vcur = ZVcur;                                            % step 9 in Algorithm 4
           Upre = Ucur;                                             % step 9 in Algorithm 4
           Vpre = Vcur;                                             % step 9 in Algorithm 4
           k = 0;                                                   % step 9 in Algorithm 4
           S = 0;
           preobj = curobj;
           preUU = Ucur;                                            % step 9 in Algorithm 4
           preVV = Vcur;                                            % step 9 in Algorithm 4
        else                                                        % step 10 in Algorithm 4
           disp(['**************Adp_RHB_NC--restart: Case 2, [',num2str(t),' ',num2str(k),'],  iter: ',num2str(iter),',  gradient: ',num2str(GRAD1(iter),15),',  preobj: ',num2str(preobj,15),',  curobj: ',num2str(curobj,15),', [B,B0]: [',num2str(BB),' ',num2str(B0),']'] );  
           Ucur = preUU;                                            % step 11 in Algorithm 4
           Vcur = preVV;                                            % step 11 in Algorithm 4
           Upre = Ucur;                                             % step 11 in Algorithm 4 
           Vpre = Vcur;                                             % step 11 in Algorithm 4
           k = 0;                                                   % step 11 in Algorithm 4
           S = 0;
           
           B0 = B0/10;                                              % step 11 in Algorithm 4 with c1=100
           rho = rho*4;                                             % equation (4) in Remark 4 with c2=2
           eta = eta/2;                                             % equation (4) in Remark 4 
           
           tt = tt+1;
           inciter1(tt) = iter;                                     % record where step 11 in Algorithm 4 is invoked
           inciter2(tt) = itercount;
           
           theta = 0.005*(eps*rho*eta^2)^(1/4);                     % reset the parameters since rho and eta change.
           KK = floor(1/theta);
           BB = sqrt(eps/rho);
        end
    end
   
    %%%%%%%%%%%%%%%%%%%%%%%%  record the information used to plot the figures  %%%%%%%%%%%%%%%%%%
    timecount = toc;
    
    iter = iter + 1;
    OBJ1(iter) = funvalue_mc(data_train,I_train,J_train,len,Ucur,Vcur);                   % objective function value at the kth iteration
    GRAD1(iter) = sqrt( norm(gradU,'fro')^2 + norm(gradV,'fro')^2 );                      % gradient norm at the kth iteration 
    itercount = itercount + 1;                         % only one gradient at (Ucur,Vcur) is computed in the implementation of the algorithm. Function values are only used to plot the figures 
    OBJ2(itercount) = OBJ1(iter);
    GRAD2(itercount) = GRAD1(iter);
        
    if iter>1
        time(iter) = time(iter-1)+timecount;
    else
        time(iter) = timecount;
    end                                                             % running time before the kth iteration
        
    if iter > N                                                     % break when it runs more than N iterations
        break;
    end
    tic
end
OBJ2 = OBJ2(1:itercount);
GRAD2 = GRAD2(1:itercount);