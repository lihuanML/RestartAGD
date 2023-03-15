clear all
format long;
%%%%%%%%%%%%%%%%%%%%%%%%   movie-10M   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% load the data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../data')
addpath('Utilities')  
dataset = 1;
namelist = {'movie-10M','movie-20M','Netflix'};             % three datasets
fname = namelist{dataset};                                  % use which dataset
load(fname,'M');
if size(M,1) > size(M,2)
    M = M';
end

normB = sqrt(sum(M.*M));
zerocolidx = find(normB==0);
if ~isempty(zerocolidx)
    fullcol = setdiff(1:size(M,2), zerocolidx);
    M = M(:,fullcol);
end                                                         % remove the 0 columns
clear fullcol
[m,n] = size(M);
[I_all,J_all,data_all] = find(M);                           % vectors of sample size
fprintf('Size is %d * %d, Sample Rate is %g \n', m, n, length(I_all)/m/n)
clear M

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

siz.m = m; siz.n = n;  siz.r = 10;                          % matrix size: m*n; rank: r

%%%%%%%%%%%%%%%%%%%%%%%%%%% run each algorithm %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note: we run the faster algorithm for 100000 iteratinos and use the minimum objective value to approximate the unknown optimal one, which will be used to plot the figures.
% Input     data_all,I_all,J_all,siz -- information of the data matrix; 
%           maxit -- iteration number; 
%           eta -- stepsize.
% Output    loss -- objective function values at all iterations
%           grad -- gradient norm at all iterations
%           time -- cumulated running time before each iteration
%           reiter -- where restart occurs
%           inciter -- where step 11 in Algorithms 2 and 4 is invoked
%           biter -- where B0 decreases smaller than B for the first time

1% our Algorithm 2
eta = 40;
[tloss1,tgrad1,loss1,grad1,time1,treiter1,tinciter1,tbiter1,reiter1,inciter1,biter1] = adp_ragd_nc(data_all,I_all,J_all,siz,100000,eta);

2% our Algorithm 4
eta = 120; 
[tloss2,tgrad2,loss2,grad2,time2,treiter2,tinciter2,tbiter2,reiter2,inciter2,biter2] = adp_rhb_nc(data_all,I_all,J_all,siz,100000,eta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% save result %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fmin = min([loss1',loss2']);
save theory_train1.mat time1 time2 loss1 loss2 grad1 grad2 tloss1 tloss2 tgrad1 tgrad2 reiter1 reiter2 treiter1 treiter2 biter1 biter2 tbiter1 tbiter2 inciter1 inciter2 tinciter1 tinciter2 fmin;

%%%%%%%%%%%%%%%%%%%%%%%%% plot the figures using the number of gradient and function evaluations as the x axis %%%%%%%%%%%%%%%%%%%%%%%

maxit = 40000;

reiter1 = reiter1(reiter1<maxit);
reiter2 = reiter2(reiter2<maxit);

figure(1);
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',20)
semilogy(1:maxit,loss2(1:maxit)-fmin,'k','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(1:maxit,loss1(1:maxit)-fmin,'r','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(reiter2,loss2(reiter2)-fmin,'ko');
hold on;
semilogy(reiter1,loss1(reiter1)-fmin,'ro');
hold on;
semilogy(inciter1,loss1(inciter1)-fmin,'o','MarkerFaceColor','c','MarkerSize',15);
hold on;
semilogy(inciter2,loss2(inciter2)-fmin,'o','MarkerFaceColor','c','MarkerSize',15);
hold on;
semilogy(biter1,loss1(biter1)-fmin,'o','MarkerFaceColor','y','MarkerSize',15);
hold on;
semilogy(biter2,loss2(biter2)-fmin,'o','MarkerFaceColor','y','MarkerSize',15);
hold on;
set(gca,'Fontsize',20)
xlabel('# of gradient and function evaluations');
ylabel('Objective error');
axis([0,40000,1e-5,10]);
set(gca,'ytick',[1e-5,1e-2,10]);
set(gca,'Fontsize',20);
legend('Ada-RHB-NC','Ada-RAGD-NC');

figure(2);
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',20)
semilogy(1:maxit,grad2(1:maxit),'k','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(1:maxit,grad1(1:maxit),'r','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(reiter2,grad2(reiter2),'ko');
hold on;
semilogy(reiter1,grad1(reiter1),'ro');
hold on;
semilogy(inciter1,grad1(inciter1),'o','MarkerFaceColor','c','MarkerSize',15);
hold on;
semilogy(inciter2,grad2(inciter2),'o','MarkerFaceColor','c','MarkerSize',15);
hold on;
semilogy(biter1,grad1(biter1),'o','MarkerFaceColor','y','MarkerSize',15);
hold on;
semilogy(biter2,grad2(biter2),'o','MarkerFaceColor','y','MarkerSize',15);
hold on;
set(gca,'Fontsize',20);
xlabel('# of gradient and function evaluations');
ylabel('Gradient norm');
axis([0,40000,1e-6,1e-1]);
set(gca,'ytick',[1e-6,1e-4,1e-1]);
set(gca,'Fontsize',20);
legend('Ada-RHB-NC','Ada-RAGD-NC');
