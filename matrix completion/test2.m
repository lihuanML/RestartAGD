clear all
format long;
%%%%%%%%%%%%%%%%%%%%%%%%   movie-20M   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% load the data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('../data')
addpath('Utilities')  
dataset = 2;                                               
namelist = {'movie-10M','movie-20M','Netflix'};            % three datasets
fname = namelist{dataset};                                 % use which dataset
load(fname,'M');
if size(M,1) > size(M,2)
    M = M';
end

normB = sqrt(sum(M.*M));
zerocolidx = find(normB==0);
if ~isempty(zerocolidx)
    fullcol = setdiff(1:size(M,2), zerocolidx);
    M = M(:,fullcol);
end                                                        % remove the 0 columns
clear fullcol
[m,n] = size(M);
[I_all,J_all,data_all] = find(M);                          % vectors of sample size
fprintf('Size is %d * %d, Sample Rate is %g \n', m, n, length(I_all)/m/n)
clear M

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

siz.m = m; siz.n = n;  siz.r = 10;        % matrix size: m*n; rank: r
maxit = 1000;                             % plot the figures using the first 1000 iterations.

%%%%%%%%%%%%%%%%%%%%%%%%%%% run each algorithm %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Note1: we tune the stepsize 'eta' for each method.
% Note2: we run some fast algorithms for 2000 iteratinos and use the minimum objective value to approximate the unknown optimal one, which will be used to plot the figures.
% Input     data_all,I_all,J_all,siz -- information of the data matrix; 
%           maxit -- iteration number; 
%           eta -- stepsize.
% Output    loss -- objective function values at all iterations
%           grad -- gradient norm at all iterations
%           time -- cumulated running time before each iteration
%           reiter -- where restart occurs
%           inciter -- where step 11 in Algorithms 2 and 4 is invoked
%           biter -- where B0 decreases smaller than B for the first time


0% gradient descent
eta = 150; % objective increases dramatically when eta=160
[tloss0,tgrad0,loss0,grad0,time0] = gd(data_all,I_all,J_all,siz,maxit,eta);

1% our Algorithm 2
eta = 50; % objective increases dramatically when eta=60
[tloss1,tgrad1,loss1,grad1,time1,treiter1,tinciter1,tbiter1,reiter1,inciter1,biter1] = adp_ragd_nc(data_all,I_all,J_all,siz,maxit*2,eta);

2% our Algorithm 4
eta = 120; % objective increases dramatically when eta=150; gradient norm decreases slower when eta=130,140
[tloss2,tgrad2,loss2,grad2,time2,treiter2,tinciter2,tbiter2,reiter2,inciter2,biter2] = adp_rhb_nc(data_all,I_all,J_all,siz,maxit*2,eta);

3% Jin's AGD
eta = 60; % eta=70: go to the first case of NCE (i.e., discard the update and go back to the previous iterate) frequently
[tloss3,tgrad3,loss3,grad3,time3] = agd_jin(data_all,I_all,J_all,siz,maxit,eta);

4% convex until proven guilty
eta = 60; % eta=70: NCE frequently, slow
[tloss4,tgrad4,loss4,grad4,time4] = guilty(data_all,I_all,J_all,siz,maxit,eta);

5% empirical AGD
eta = 40; % objective increases dramatically when eta=50
[tloss5,tgrad5,loss5,grad5,time5,treiter5,reiter5] = emp_ragd(data_all,I_all,J_all,siz,maxit*2,eta);

6% conjugate gradient descent
eta = 100; % no effect due to line search
[tloss6,tgrad6,loss6,grad6,time6] = cg(data_all,I_all,J_all,siz,maxit*2,eta);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% save result %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fmin = min([loss0',loss1',loss2',loss3',loss4,loss5',loss6']);
save result_train2.mat time0 time1 time2 time3 time4 time5 time6 loss0 loss1 loss2 loss3 loss4 loss5 loss6 grad0 grad1 grad2 grad3 grad4 grad5 grad6 tloss0 tloss1 tloss2 tloss3 tloss4 tloss5 tloss6 tgrad0 tgrad1 tgrad2 tgrad3 tgrad4 tgrad5 tgrad6 reiter1 reiter2 treiter1 treiter2 reiter5 treiter5 biter1 biter2 tbiter1 tbiter2 inciter1 inciter2 tinciter1 tinciter2 fmin;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% plot the figures using time as the x axis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

treiter1 = treiter1(treiter1<maxit);
treiter2 = treiter2(treiter2<maxit);                 % only plot maxit iterations

figure(1);
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',20)
semilogy(time0(1:maxit),tloss0(1:maxit)-fmin,'color',[.7,.3,0],'LineWidth',1,'MarkerSize',10);
hold on;
semilogy(time4(1:maxit),tloss4(1:maxit)-fmin,'g','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(time3(1:maxit),tloss3(1:maxit)-fmin,'m','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(time5(1:maxit),tloss5(1:maxit)-fmin,'b','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(time6(1:maxit),tloss6(1:maxit)-fmin,'c','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(time1(1:maxit),tloss1(1:maxit)-fmin,'r','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(time2(1:maxit),tloss2(1:maxit)-fmin,'k','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(time1(treiter1),tloss1(treiter1)-fmin,'o','MarkerFaceColor','r');
hold on;
semilogy(time2(treiter2),tloss2(treiter2)-fmin,'o','MarkerFaceColor','k');
hold on;
set(gca,'Fontsize',20)
xlabel('Time (s)');
ylabel('Objective error');
%axis([0,1500,0,1.5]);
set(gca,'Fontsize',20);
legend('GD','Convex-until-proven-guilty','AGD-Jin','Heuristic RAGD','Conjugate Gradient','Ada-RAGD-NC','Ada-RHB-NC');

figure(2);
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',20)
semilogy(time0(1:maxit),tgrad0(1:maxit),'color',[.7,.3,0],'LineWidth',1,'MarkerSize',10);
hold on;
semilogy(time6(1:maxit),tgrad6(1:maxit),'c','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(time4(1:maxit),tgrad4(1:maxit),'g','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(time3(1:maxit),tgrad3(1:maxit),'m','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(time5(1:maxit),tgrad5(1:maxit),'b','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(time1(1:maxit),tgrad1(1:maxit),'r','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(time2(1:maxit),tgrad2(1:maxit),'k','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(time1(treiter1),tgrad1(treiter1),'o','MarkerFaceColor','r');
hold on;
semilogy(time2(treiter2),tgrad2(treiter2),'o','MarkerFaceColor','k');
hold on;
set(gca,'Fontsize',20);
xlabel('Time (s)');
ylabel('Gradient norm');
%axis([0,1500,1e-5,1e-1]);
set(gca,'Fontsize',20);
legend('GD','Conjugate Gradient','Convex-until-proven-guilty','AGD-Jin','Heuristic RAGD','Ada-RAGD-NC','Ada-RHB-NC');

%%%%%%%%%%%%%%%%%%%%%%%%% plot the figures using the number of gradient and function evaluations as the x axis %%%%%%%%%%%%%%%%%%%%%%%

reiter1 = reiter1(reiter1<floor(length(loss1)/2));
reiter2 = reiter2(reiter2<floor(length(loss2)/2));

figure(3);
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',20)
semilogy(loss0-fmin,'color',[.7,.3,0],'LineWidth',1,'MarkerSize',10);
hold on;
semilogy(loss4-fmin,'g','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(loss3-fmin,'m','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(loss6(1:floor(length(loss6)/2))-fmin,'c','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(loss5(1:floor(length(loss5)/2))-fmin,'b','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(loss1(1:floor(length(loss1)/2))-fmin,'r','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(loss2(1:floor(length(loss2)/2))-fmin,'k','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(reiter1,loss1(reiter1)-fmin,'o','MarkerFaceColor','r');
hold on;
semilogy(reiter2,loss2(reiter2)-fmin,'o','MarkerFaceColor','k');
hold on;
set(gca,'Fontsize',20)
xlabel('# of gradient and function evaluations');
ylabel('Objective error');
%axis([0,1500,0,1.5]);
set(gca,'Fontsize',20);
legend('GD','Convex-until-proven-guilty','AGD-Jin','Conjugate Gradient','Heuristic RAGD','Ada-RAGD-NC','Ada-RHB-NC');

figure(4);
set (gcf,'Position',[1,1,700,600], 'color','w')
set(gca,'Fontsize',20)
semilogy(grad0,'color',[.7,.3,0],'LineWidth',1,'MarkerSize',10);
hold on;
semilogy(grad6(1:floor(length(grad6)/2)),'c','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(grad4,'g','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(grad3,'m','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(grad5(1:floor(length(grad5)/2)),'b','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(grad1(1:floor(length(grad1)/2)),'r','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(grad2(1:floor(length(grad2)/2)),'k','LineWidth',1,'MarkerSize',10);
hold on;
semilogy(reiter1,grad1(reiter1),'o','MarkerFaceColor','r');
hold on;
semilogy(reiter2,grad2(reiter2),'o','MarkerFaceColor','k');
hold on;
set(gca,'Fontsize',20);
xlabel('# of gradient and function evaluations');
ylabel('Gradient norm');
%axis([0,1500,1e-5,1e-1]);
set(gca,'Fontsize',20);
legend('GD','Conjugate Gradient','Convex-until-proven-guilty','AGD-Jin','Heuristic RAGD','Ada-RAGD-NC','Ada-RHB-NC');
