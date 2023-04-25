function [OBJ1,GRAD1,OBJ2,GRAD2,time] = agd_jin(data_train,I_train,J_train,siz,N,eta)

tic

m = siz.m; n = siz.n;  r = siz.r; 
len = length(I_train);                                     % sample size
Mat = sparse(I_train,J_train,data_train,m,n);              % a sparse matrix used to compute the gradient efficiently  

X = sparse(I_train,J_train,data_train,m,n);
[A,Sigma,B] = svds(X,r);
Sigma = sqrt(Sigma);
Ucur = A*Sigma;                      % initializaton using the SVD of the observed sparse matrix
Vcur = B*Sigma;                      % U; m*r; V: n*r

time = zeros(N,1);                   % running time
GRAD1 = zeros(N,1);                  % gradient norm; plot the figures using time as the x axis
OBJ1 = zeros(N,1);                   % objective function value; plot the figures using time as the x axis
GRAD2 = zeros(N*4,1);                % gradient norm; plot the figures using the number of gradient and function evaluations as the x axis
OBJ2 = zeros(N*4,1);                 % objective function value; plot the figures using the number of gradient and function evaluations as the x axis
    
DiffU = zeros(m,r);                  % Ucur-Upre
DiffV = zeros(n,r);
    
eps = 1e-4;                          % precison
rho = 1;                             % Hessian Lipschitz constant
theta = 0.04*(eps*rho*eta^2)^(1/4);  % (3) in Jin2018 suggests theta = 0.5*(eps*rho*eta^2)^(1/4).
                                     % we tune it for better performance on Movie10 dataset; 0.04*(eps*rho*eta^2)^(1/4)=0.025
gamma = theta^2/eta;                 % parameter in Jin2018. set it following (3) in Jin2018
s = gamma/rho/4;                     % parameter in Jin2018. set it following (3) in Jin2018
itercount = 0;                       % number of gradient and function evaluations used to plot the figures
[theta,gamma,s]

for k = 1:N                          % run N iterations

    %%%%%%%%%%%%%%%%%% AGD step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    TUcur = Ucur+(1-theta)*DiffU;
    TVcur = Vcur+(1-theta)*DiffV;

    [gradU,gradV,obj] = grad_funvalue_mc(data_train,I_train,J_train,Mat,len,TUcur,TVcur);
    % calculate the gradient and functino value at (TUcur,TVcur)

    Upre = Ucur;
    Vpre = Vcur;

    Ucur = TUcur-eta*gradU;
    Vcur = TVcur-eta*gradV;                                    % gradient descent step at (TUcur,TVcur)

    DiffUpre = DiffU;
    DiffU =  Ucur - Upre;                                      % momentum
    DiffVpre = DiffV;
    DiffV =  Vcur - Vpre;
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NCE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fy = obj;                                                                          % calculate the objective function value at (TUcur,TVcur)
    fx = funvalue_mc(data_train,I_train,J_train,len,Upre,Vpre);                        % calculate the objective function value at (Ucur,Vcur)

    OBJ1(k) = fx;                                                                      % objective function value at the kth iteration
    
    if fx < fy + sum(sum(gradU.*(Upre-TUcur))) + sum(sum(gradV.*(Vpre-TVcur))) - gamma/2*norm(Upre-TUcur,'fro')^2 - gamma/2*norm(Vpre-TVcur,'fro')^2     %  check NCE condition
      if sqrt( norm(DiffUpre,'fro')^2+norm(DiffVpre,'fro')^2 ) >= s                    % case 1
          % after case 1, the next AGD step reduces to one GD step
          Ucur = Upre;
          Vcur = Vpre;
          disp(['JAGD--NCE--first case'] );
      else                                                                             % case 2
          % for high dimension problems, norm(DiffUpre,'fro') and norm(DiffVpre,'fro') are not likely to be small. 
          % So case 2 is almost never triggered
          Udelta = s*DiffUpre/sqrt(norm(DiffUpre,'fro')^2+norm(DiffVpre,'fro')^2);
          Vdelta = s*DiffVpre/sqrt(norm(DiffUpre,'fro')^2+norm(DiffVpre,'fro')^2);     % negative curvature direction
 
          Ucur1 = Upre + Udelta;
          Vcur1 = Vpre + Vdelta;
          f1 = funvalue_mc(data_train,I_train,J_train,len,Ucur1,Vcur1);
          
          Ucur2 = Upre - Udelta;
          Vcur2 = Vpre - Vdelta;
          f2 = funvalue_mc(data_train,I_train,J_train,len,Ucur2,Vcur2);
       
          if f1<f2                                                                      % choose the one with smaller function value
              Ucur = Ucur1;
              Vcur = Vcur1;
              OBJ1(k) = f1; 
          else
              Ucur = Ucur2;
              Vcur = Vcur2;
              OBJ1(k) = f2; 
          end
    
          itercount = itercount + 1;                             % one function value at (Ucur1,Vcur1)
          OBJ2(itercount) = OBJ1(k);
          GRAD2(itercount) = sqrt( norm(gradU,'fro')^2 + norm(gradV,'fro')^2 );
          itercount = itercount + 1;                             % one function value at (Ucur2,Vcur2)
          OBJ2(itercount) = OBJ1(k);
          GRAD2(itercount) = sqrt( norm(gradU,'fro')^2 + norm(gradV,'fro')^2 );
          
          disp(['JAGD--NCE--second case'] );
          
      end
      DiffU = zeros(m,r);
      DiffV = zeros(n,r);
      disp(['JAGD--NCE--iter: ',num2str(k)] );
    end
   
    %%%%%%%%%%%%%%%%%% record the information used to plot the figures  %%%%%%%%%%%%%%%%%%
    
    timecount = toc;
    
    GRAD1(k) = sqrt( norm(gradU,'fro')^2 + norm(gradV,'fro')^2 );      % gradient norm at the kth iteration 
    itercount = itercount + 1;                                         % one gradient at (TUcur,TVcur)
    OBJ2(itercount) = OBJ1(k);
    GRAD2(itercount) = GRAD1(k);
    itercount = itercount + 1;                                         % one function value at (TUcur,TVcur)
    OBJ2(itercount) = OBJ1(k);
    GRAD2(itercount) = GRAD1(k);
    itercount = itercount + 1;                                         % one function value at (Upre,Vpre)
    OBJ2(itercount) = OBJ1(k);
    GRAD2(itercount) = GRAD1(k);
   
    if k>1
        time(k) = time(k-1)+timecount;
    else
        time(k) = timecount;
    end                                                                % running time before the kth iteration
          
    if mod(k,100) == 0                                                 % print every 100 iterations
        disp([ 'AGD-Jin--iter: ',num2str(k),', gradient: ',num2str(GRAD1(k),15),', obj: ',num2str(OBJ1(k),15) ] );
    end
    tic
end
OBJ2 = OBJ2(1:itercount);
GRAD2 = GRAD2(1:itercount);