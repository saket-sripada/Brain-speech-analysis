%load('S02thtjaw1.mat')
%load('S02thtjaw2.mat')
%load('S02thtjaw3.mat')
%load('S02thtjaw4.mat')
load('S02thtjaw5.mat')
load('S03meg2jaw2speech_p1.mat')
load('S03meg2jaw2speech_p3.mat')
load('S03meg2jaw2speech_p4.mat')
load('S03meg2jaw2speech_p5.mat')
%%
% Data prep
%Xsimple = [ S3mph1{1,1}; S3mph3{1,1}; S3mph4{1,1}; S3mph5{1,1} ]; 
%ysimple = [ S3jphr1(:,1); S3jphr3(:,1); S3jphr4(:,1); S3jphr5(:,1) ];
X = S3mph1{1,1};
y = S3jphr1(:,1);
X = [ X ; S3mph3{1,1} ];
y = [ y ; S3jphr3(:,1)];
X = [ X ; S3mph4{1,1} ];
y = [ y ; S3jphr4(:,1)];
X = [ X ; S3mph5{1,1} ];
y = [ y ; S3jphr5(:,1)];
%z = speech1(2501:4500,1);
for i = 2:size(S3mph1,1)
    X = [ X ; S3mph1{i,1} ];
    y = [ y ; S3jphr1(:,i) ];
end
for i = 1:size(S3mph3,1)
    X = [ X ; S3mph3{i,1} ];
    y = [ y ; S3jphr3(:,i) ];
end
for i = 1:size(S3mph4,1)
    X = [ X ; S3mph3{i,1} ];
    y = [ y ; S3jphr3(:,i) ];
end
for i = 1:size(S3mph5,1)
    X = [ X ; S3mph3{i,1} ];
    y = [ y ; S3jphr3(:,i) ];
end
%%
% Cross varidation (train: 70%, test: 30%)
cv = cvpartition(size(X,1),'HoldOut',0.3);
% %cvy = cvpartition(size(y,1),'HoldOut',0.3);
% idx = cv.test;
% % Separate to training and test data
% Xtrn = X(~idx,:);
% yTrn = y(~idx,:);
% Xval = X(idx,:);
% yVal = y(idx,:);
%%
% model building and testing
%mod = fitrkernel(X,y,'kFold',5);
[mod,FitInfo,HyperparameterOptimizationResults] = fitrkernel(X,y,'OptimizeHyperparameters','auto',...
   'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'))
loloss = loss(mod,X,y)
save('optkernel',mod)
% [B,FitInfo] = lasso(X,y,'CV',10)
% lassoPlot(B,FitInfo,'PlotType','CV');
% legend('show') % Show legend
MAE = 0;
for i = 1:size(jphr5,2)
    Xtst = mph5{i,1};
    yTst = jphr5(:,i);
    yPred = predict(mod,Xtst);
    e = mae(yPred - yTst);
       e = sqrt(mean((yPred - yTst).^2));
    MAE = MAE + e ;
    pltvec = [yTst yPred];
    figure
    plot(pltvec,'DisplayName','pltvec');
    title(['trial ', num2str(i),' : MAE = ',num2str(e)]);
    %savefig(num2str(i));
    print(num2str(i),'-dpng');
end
avgMAE = MAE/size(jphr5,2)