%%
% load training data
load('S03meg2jaw2speech_p1.mat')
load('S03meg2jaw2speech_p3.mat')
load('S03meg2jaw2speech_p4.mat')
load('S03meg2jaw2speech_p5.mat')
X = S3mph1{1,1};
y = S3jphr1(:,1);
for i = 2:size(mph1,1)
    X = [ X ; mph1{i,1} ];
    y = [ y ; jphr1(:,i) ];
end
for i = 1:size(mph3,1)
    X = [ X ; mph3{i,1} ];
    y = [ y ; jphr3(:,i) ];
end
for i = 1:size(mph4,1)
    X = [ X ; mph3{i,1} ];
    y = [ y ; jphr3(:,i) ];
end
for i = 1:size(mph5,1)
    X = [ X ; mph3{i,1} ];
    y = [ y ; jphr3(:,i) ];
end
clear mph1;
clear mph3;
clear mph4;
clear mph5;
clear jphr1;
clear jphr3;
clear jphr4;
clear jphr5;
%%
% Cross varidation (train: 70%, test: 30%)
% cv = cvpartition(size(X,1),'HoldOut',0.3);
% %cvy = cvpartition(size(y,1),'HoldOut',0.3);
% idx = cv.test;
% % Separate to training and test data
% Xtrn = X(~idx,:);
% yTrn = y(~idx,:);
% Xval = X(idx,:);
% yVal = y(idx,:);
%%
%load testing data

%load('S2thtjaw1.mat')
%load('S2thtjaw2.mat')
%load('S2thtjaw3.mat')
%load('S2thtjaw4.mat')
load('S2thtjaw5.mat')
%%
% model building and testing
%mod = fitrkernel(Xtrn,yTrn,'kFold',5);
[mod,FitInfo,HyperparameterOptimizationResults] = fitrkernel(Xtrn,yTrn,'OptimizeHyperparameters','auto',...
   'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'))
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