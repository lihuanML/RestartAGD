function [OBJ1,GRAD1,OBJ2,GRAD2,TIME] = guilty(data_train,I_train,J_train,siz,N,eta)
% We implement the algorithm in the main text of Carmon2017. We omit the modifications in their Section D because it is hard to replicate
    tic 
    
    m = siz.m; n = siz.n;  r = siz.r; 
    len = length(I_train);                                 % sample size
    Mat = sparse(I_train,J_train,data_train,m,n);          % a sparse matrix used to compute the gradient efficiently  

    X = sparse(I_train,J_train,data_train,m,n);
    [A,Sigma,B] = svds(X,r);
    Sigma = sqrt(Sigma);
    PUcur = A*Sigma;                      % initializaton using the SVD of the observed sparse matrix
    PVcur = B*Sigma;                      % U; m*r; V: n*r

    TIME = [];                   % running time
    GRAD1 = [];                  % gradient norm; plot the figures using time as the x axis
    OBJ1 = [];                   % objective function value; plot the figures using time as the x axis
    GRAD2 = [];                  % gradient norm; plot the figures using the number of gradient and function evaluations as the x axis
    OBJ2 = [];                   % objective function value; plot the figures using the number of gradient and function evaluations as the x axis
    time_end = 0;
    
    eps = 1e-14;                 % precison. function value decreases slower and gradient norm oscilates in a larger range when eps=1e-12
    rho = 1;                     % Hessian Lipschitz constant
    NN = N/10;                   % maximum iteration number in each inner loop. Otherwise, the inner loop may not break
    mu = 2*sqrt(rho*eps);        % alpha in  Guarded-non-convex-AGD in Carmon2017. Set it following the proof of Theorem 1 in Carmon2017
    
    for k = 1:N        

        [Ucur,Vcur,Umin,Vmin,fmin,Unce1,Unce2,Vnce1,Vnce2,sign,obj1,grad1,obj2,grad2,time] = AGD_until_guilty(data_train,I_train,J_train,Mat,PUcur,PVcur,NN,eta,mu,len,eps/10);
        %  AGD-until-proven-guilty in Carmon2017
        if sign == 0             % line 4 in  Guarded-non-convex-AGD in Carmon2017 
            PUcur = Ucur;
            PVcur = Vcur;
            
            disp([ 'Guilty--iter: Convex AGD, outer: ',num2str(k)] );
        else
            s = mu/rho;          % eta in  Exploit-NC-pair. set it following the proof of Lemma 3 in Carmon2017
            [Umin2,Vmin2,fmin2] = NCE(data_train,I_train,J_train,Unce1,Unce2,Vnce1,Vnce2,s,len);         %  Exploit-NC-pair in Carmon2017
            % [Umin,Vmin,fmin] is returned by Find-best-iterate embedded in AGD_until_guilty
            if fmin<fmin2
                PUcur = Umin;
                PVcur = Vmin;
            else
                PUcur = Umin2;
                PVcur = Vmin2;
            end
            
            disp([ 'Guilty--iter: NCE, outer: ',num2str(k), '  ',num2str(length(TIME))] );
        end
        
        GRAD1 = [GRAD1,grad1];
        OBJ1 = [OBJ1,obj1];
        GRAD2 = [GRAD2,grad2];
        OBJ2 = [OBJ2,obj2];
        TIME = [TIME,time+time_end];
        time_end = time(end)+time_end;
            
        if length(TIME)>=N
            break;
        end                      % break when running N iterations
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [Ucur,Vcur,Umin,Vmin,fmin,Unce1,Unce2,Vnce1,Vnce2,sign,obj1,GRAD1,obj2,GRAD2,time] = AGD_until_guilty(data_train,I_train,J_train,Mat,Ucur0,Vcur0,NN,eta,mu,len,eps)
%  AGD-until-proven-guilty in Carmon2017

    %%%%%%%%%%%%%%%%%%%%%%%%% initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    TUcur = Ucur0;
    TVcur = Vcur0;
    Ucur = Ucur0;
    Vcur = Vcur0;
    My = partXY(Ucur0',Vcur0',I_train,J_train,len);                                         % calculate Proj(U*V') and store it in the vector M, where Proj means the projection onto the index set of the observed samples
    % denote f(x)=e^x/(1+e^x)
    % denote F2(x)=-F1(x)=log f(x) if y=1 and F2(x)=-F1(x)=log(1-f(x)) if y=-1
    fyy = 1 ./ (1 + exp(-My));                                                              % f(x)=e^x/(1+e^x)=1/(1+e^{-x})
    ttlog = log( data_train.*fyy' - (data_train-1)/2 );                                     % F2(x)=log f(x) if y=1 and F2(x)=log(1-f(x)) if y=-1
    ttlog(ttlog==-inf & data_train==1) = My(ttlog==-inf & data_train==1);                   % when x approx -inf, log f(x) approx -inf. We let log f(x) approx x to avoid inf because log(exp(-1000))=-inf in matlab
    ttlog(ttlog==-inf & data_train==-1) = -My(ttlog==-inf & data_train==-1);                % when x approx inf, log(1-f(x)) approx -inf. We let log(1-f(x)) approx -x to avoid inf 
    f0 = ( -sum( ttlog ) + norm((Ucur0'*Ucur0 - Vcur0'*Vcur0),'fro')^2/2 )/len;             % initialized function value

    kappa = 1/eta/mu;                                               % condition number
    omega = ( sqrt(kappa) - 1 )/( sqrt(kappa) + 1 );                % momentum parameter

    fmin = f0;
    Umin = Ucur;
    Vmin = Vcur;

    time = [];
    GRAD1 = [];
    obj1 = [];
    GRAD2 = [];
    obj2 = [];
    sign = 0;
    itercount = 0;                                                  % number of gradient and function evaluations used to plot the figures

    Unce1 = Ucur;
    Unce2 = Ucur;
    Vnce1 = Vcur;
    Vnce2 = Vcur;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% AGD iterations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for t=1:NN                                                      % maximum 100 iterations in the inner loop 
 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% one AGD step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % denote f(x)=e^x/(1+e^x)
        % denote F1(x)=-log f(x)=-x+log(1+e^x) if y=1 and F1(x)=-log(1-f(x))=log(1+e^x) if y=-1
        % then nabla F1(x)=-1+e^x/(1+e^x)=-1/(1+e^x) if y=1 and nabla F1(x)=e^x/(1+e^x) if y=-1
        G = -1./(1+exp(My')) - (data_train-1)/2;                    % G is nabla F1(x), My is x, and data_train is y (1 or -1)
        updateSval1(Mat, G', len);                                  % recover the sparse matrix Mat from the vector G
                                                                    % That is, Mat = sparse(I_train,J_train,G,m,n). But the matlab code is much slower than the C code. 
        gradU = ( Mat * TVcur + TUcur*(TUcur'*TUcur - TVcur'*TVcur) )/len + 2*mu*(TUcur-Ucur0);
        gradV = ( Mat' * TUcur + TVcur*(TVcur'*TVcur - TUcur'*TUcur) )/len + 2*mu*(TVcur-Vcur0);        % gradient of the regularized function at (TUcur,TVcur)

        Upre = Ucur;
        Vpre = Vcur;

        Ucur = TUcur-eta*gradU;
        Vcur = TVcur-eta*gradV;

        TUcur = Ucur+omega*(Ucur-Upre);
        TVcur = Vcur+omega*(Vcur-Vpre);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% check guilty %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        Mx = partXY(Ucur',Vcur',I_train,J_train,len); 
        fxx = 1 ./ (1 + exp(-Mx));
        ttlog = log( data_train.*fxx' - (data_train-1)/2 );
        ttlog(ttlog==-inf & data_train==1) = Mx(ttlog==-inf & data_train==1);
        ttlog(ttlog==-inf & data_train==-1) = -Mx(ttlog==-inf & data_train==-1); 
        ffx = ( -sum( ttlog ) + norm((Ucur'*Ucur - Vcur'*Vcur),'fro')^2/2 )/len;
        fx = ffx + mu*norm(Ucur-Ucur0,'fro')^2 + mu*norm(Vcur-Vcur0,'fro')^2;
        % objective value of the regularized function at (Ucur,Vcur)
        My = partXY(TUcur',TVcur',I_train,J_train,len); 
        fyy = 1 ./ (1 + exp(-My));
        ttlog = log( data_train.*fyy' - (data_train-1)/2 );
        ttlog(ttlog==-inf & data_train==1) = My(ttlog==-inf & data_train==1);
        ttlog(ttlog==-inf & data_train==-1) = -My(ttlog==-inf & data_train==-1); 
        fy = ( -sum( ttlog ) + norm((TUcur'*TUcur - TVcur'*TVcur),'fro')^2/2 )/len + mu*norm(TUcur-Ucur0,'fro')^2 + mu*norm(TVcur-Vcur0,'fro')^2;
        % objective value of the regularized function at (TUcur,TVcur)
        
        [Uw,Vw,fw,sign] = Certify_Progress(data_train,I_train,J_train,Mat,Mx,Ucur,Vcur,Ucur0,Vcur0,eta,mu,kappa,t,f0,fx,len);
        %  Certify-progress in Carmon2017
        if sign == 1                 % do not return null
            % run Find_witness_pair in Carmon2017
            % we only compare (Uw,Vw) and (Ucur,Vcur) with current (TUcur,TVcur) and drop the the previous (TUcur,TVcur) because we do not store them
            if fx < fy + sum(sum(gradU.*(Ucur-TUcur))) + sum(sum(gradV.*(Vcur-TVcur))) + mu/2*norm(Ucur-TUcur,'fro')^2 + mu/2*norm(Vcur-TVcur,'fro')^2
                Unce1 = Ucur;
                Unce2 = TUcur;
                Vnce1 = Vcur;
                Vnce2 = TVcur;
            end
            if fw < fy + sum(sum(gradU.*(Uw-TUcur))) + sum(sum(gradV.*(Vw-TVcur))) + mu/2*norm(Uw-TUcur,'fro')^2 + mu/2*norm(Vw-TVcur,'fro')^2
                Unce1 = Uw;
                Unce2 = TUcur;
                Vnce1 = Vw;
                Vnce2 = TVcur;
            end
        end
    
        %%%%%%%%%%%%%%%%%% record the information used to plot the figures  %%%%%%%%%%%%%%%%%%
        
        timecount = toc;
        
        obj1(t) = ffx;
        GRAD1(t) = sqrt( norm(gradU,'fro')^2 + norm(gradV,'fro')^2 );
        
        itercount = itercount + 1;                 % one gradient at (TUcur,TVcur)
        obj2(itercount) = ffx;
        GRAD2(itercount) = GRAD1(t);
        itercount = itercount + 1;                 % one function value at (Ucur,Vcur)   
        obj2(itercount) = ffx;
        GRAD2(itercount) = GRAD1(t);
        itercount = itercount + 1;                 % one function value at (TUcur,TVcur)
        obj2(itercount) = ffx;
        GRAD2(itercount) = GRAD1(t);
        if fx<=f0
            itercount = itercount + 1;             % one gradinet at (Ucur,Vcur) in Certify_Progress         
            obj2(itercount) = ffx;
            GRAD2(itercount) = GRAD1(t);
            itercount = itercount + 1;             % one function value at (Uw,Vw) in Certify_Progress
            obj2(itercount) = ffx;
            GRAD2(itercount) = GRAD1(t);
        end
        
        if ffx<fmin                                % Find-best-iterate in Carmon2017
           fmin = ffx;
           Umin = Ucur;
           Vmin = Vcur;
        end
        
        if t>1
            time(t) = time(t-1)+timecount;
        else
            time(t) = toc;
        end
        
        if mod(t,100) == 0
            disp([ 'Guilty--iter: inner  ',num2str(t),', gradient: ',num2str(GRAD1(t),15),', obj: ',num2str(obj1(t),15) ] );
        end
        
        tic
        
        if sign == 1
            break;
        end                                        % line 8 in AGD-until-proven-guilty in Carmon2017
        if GRAD1(t)<eps
            break;
        end        
        
    end
end

function [Uw,Vw,fw,sign] = Certify_Progress(data_train,I_train,J_train,Mat,Mx,Ucur,Vcur,Ucur0,Vcur0,eta,mu,kappa,k,f0,fx,len)

    if fx>f0
        sign = 1;
        Uw = Ucur0;
        Vw = Vcur0;
        fw = f0;
    else

        G = -1./(1+exp(Mx')) - (data_train-1)/2;
        updateSval1(Mat, G', len);
        
        gradU = ( Mat * Vcur + Ucur*(Ucur'*Ucur - Vcur'*Vcur) )/len + 2*mu*(Ucur-Ucur0);
        gradV = ( Mat' * Ucur + Vcur*(Vcur'*Vcur - Ucur'*Ucur) )/len + 2*mu*(Vcur-Vcur0);          % gradient of the regularized function at (Ucur,Vcur)
        gradx = sqrt( norm(gradU,'fro')^2 + norm(gradV,'fro')^2 );                                 % gradient norm

        Uw = Ucur-eta*gradU;
        Vw = Vcur-eta*gradV;                                                                       % one GD step

        Mw = partXY(Uw',Vw',I_train,J_train,len); 
        fww = 1 ./ (1 + exp(-Mw));
        ttlog = log( data_train.*fww' - (data_train-1)/2 );
        ttlog(ttlog==-inf & data_train==1) = Mw(ttlog==-inf & data_train==1);
        ttlog(ttlog==-inf & data_train==-1) = -Mw(ttlog==-inf & data_train==-1); 
        fw = ( -sum( ttlog ) + norm((Uw'*Uw - Vw'*Vw),'fro')^2/2 )/len + mu*norm(Uw-Ucur0,'fro')^2 + mu*norm(Vw-Vcur0,'fro')^2;
        % objective value of the regularized function at (Uw,Vw)
        
        if gradx^2 > 2/eta * (1-1/sqrt(kappa))^k * ( f0-fw+mu/2*(norm(Uw-Ucur0,'fro')^2+norm(Vw-Vcur0,'fro')^2) )
            sign = 1;
        else
            sign = 0;                                                                              % return null
        end
    end

end
    

function [Umin,Vmin,fmin] = NCE(data_train,I_train,J_train,Unce1,Unce2,Vnce1,Vnce2,s,len)          %  Exploit-NC-pair in Carmon2017

    Udelta = s * (Unce1-Unce2)/sqrt(norm(Unce1-Unce2,'fro')^2+norm(Vnce1-Vnce2,'fro')^2);
    Vdelta = s * (Vnce1-Vnce2)/sqrt(norm(Unce1-Unce2,'fro')^2+norm(Vnce1-Vnce2,'fro')^2);

    Ucur1 = Unce1 + Udelta;
    Vcur1 = Vnce1 + Vdelta;
    M1 = partXY(Ucur1',Vcur1',I_train,J_train,len); 
    f = 1 ./ (1 + exp(-M1));
    ttlog = log( data_train.*f' - (data_train-1)/2 );
    ttlog(ttlog==-inf & data_train==1) = M1(ttlog==-inf & data_train==1);
    ttlog(ttlog==-inf & data_train==-1) = -M1(ttlog==-inf & data_train==-1); 
    f1 = ( -sum( ttlog ) + norm((Ucur1'*Ucur1 - Vcur1'*Vcur1),'fro')^2/2 )/len;
  
    Ucur2 = Unce1 - Udelta;
    Vcur2 = Vnce1 - Vdelta;
    M2 = partXY(Ucur2',Vcur2',I_train,J_train,len); 
    f = 1 ./ (1 + exp(-M2));
    ttlog = log( data_train.*f' - (data_train-1)/2 );
    ttlog(ttlog==-inf & data_train==1) = M2(ttlog==-inf & data_train==1);
    ttlog(ttlog==-inf & data_train==-1) = -M2(ttlog==-inf & data_train==-1); 
    f2 = ( -sum( ttlog ) + norm((Ucur2'*Ucur2 - Vcur2'*Vcur2),'fro')^2/2 )/len;

    if f1<f2                                    % choose the one with smaller function value
      Umin = Ucur1;
      Vmin = Vcur1;
      fmin = f1;
    else
      Umin = Ucur2;
      Vmin = Vcur2;
      fmin = f2;
    end
  
end
