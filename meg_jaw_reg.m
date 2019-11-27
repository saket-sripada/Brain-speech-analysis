%% 
%S02
%jawS2 = load('D:\saket\thought-jaw\S02_pf_Jaw.mat');
%megS2 =load('D:\saket\thought-jaw\S02_pf_speech.mat');

% jphr1 = jawS2.data_pf_1(2501:4000,:);
% jphr2 = jawS2.data_pf_2(2501:4000,:);
% jphr3 = jawS2.data_pf_3(2501:4000,:);
% jphr4 = jawS2.data_pf_4(2501:4000,:);
% jphr5 = jawS2.data_pf_5(2501:4000,:);
% mph1 = megS2.pf_speech_1.data;
% mph2 = megS2.pf_speech_2.data;
% mph3 = megS2.pf_speech_3.data;
% mph4 = megS2.pf_speech_4.data;
% mph5 = megS2.pf_speech_5.data;
clear megS2;
clear jawS2;
%%
%S01
% jawS1 = load('D:\saket\thought-jaw\S01_Jaw.mat');
% megS1 =load('D:\saket\thought-jaw\Wang_speech_1.mat');

% jphr1 = jawS1.data_jw_1(2501:4000,:);
% jphr2 = jawS1.data_jw_2(2501:4000,:);
% jphr3 = jawS1.data_jw_4(2501:4000,:);
% jphr4 = jawS1.data_jw_8(2501:4000,:);
% jphr5 = jawS1.data_jw_16(2501:4000,:);
% mph1 = megS1.data_1.data;
% mph2 = megS1.data_2.data;
% mph3 = megS1.data_4.data;
% mph4 = megS1.data_8.data;
% mph5 = megS1.data_16.data;
clear megS1;
clear jawS1;
%%
%S03
jawS3 = load('D:\saket\thought-jaw\S03_sr_Jaw.mat');
megS3 =load('D:\saket\thought-jaw\S03_sr_speech.mat');

jphr1 = jawS3.data_sr_1(2501:4000,:);
jphr2 = jawS3.data_sr_2(2501:4000,:);
jphr3 = jawS3.data_sr_3(2501:4000,:);
jphr4 = jawS3.data_sr_4(2501:4000,:);
jphr5 = jawS3.data_sr_5(2501:4000,:);
mph1 = megS3.sr_speech_1.data;
mph2 = megS3.sr_speech_2.data;
mph3 = megS3.sr_speech_3.data;
mph4 = megS3.sr_speech_4.data;
mph5 = megS3.sr_speech_5.data;
clear megS3;
clear jawS3;
%%
megshape=size(mph1)
for i = 1:megshape(1,1)
    mph1{i,1} = mph1{i,1}(2501:4000,110:313);
end
megshape=size(mph2)
for i = 1:megshape(1,1)
    mph2{i,1} = mph2{i,1}(2501:4000,108:311);
end
megshape=size(mph3)
for i = 1:megshape(1,1)
    mph3{i,1} = mph3{i,1}(2501:4000,108:311);
end
megshape=size(mph4)
for i = 1:megshape(1,1)
    mph4{i,1} = mph4{i,1}(:,2:205);
end
megshape=size(mph5)
for i = 1:megshape(1,1)
    mph5{i,1} = mph5{i,1}(2501:4000,110:313);
end
%%
% Cross varidation (train: 70%, test: 30%)
X = mph1{1,1};
cv = cvpartition(size(X,1),'HoldOut',0.3);
idx = cv.test;
% Separate to training and test data
Xtrn = X(~idx,:);
Xtst  = X(idx,:);

% Cross varidation (train: 70%, test: 30%)
y = jphr1(:,1)
cv = cvpartition(size(y,1),'HoldOut',0.3);
idx = cv.test;
% Separate to training and test data
yTrn = y(~idx,:);
yTst  = y(idx,:);
%%
%LSTM

% inputSize = 200;
% numHiddenUnits = 64;
% 
% LSTMmodel = [ ...
%     sequenceInputLayer(inputSize)
%     lstmLayer(numHiddenUnits)
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer]
%X = [mph1{i,1};
%[XTrain,YTrain] = prepareDataTrain(filenamePredictors);
% z-normalising
% mu = mean([XTrain{:}],2);
% sig = std([XTrain{:}],0,2);
% 
% for i = 1:numel(XTrain)
%     XTrain{i} = (XTrain{i} - mu) ./ sig;
% end

numResponses = size(yTrn,1)
featureDimension = size(Xtrn,1)
numHiddenUnits = 256;

layers = [ ...
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(128)
    dropoutLayer(0.5)
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 60;
miniBatchSize = 20;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'Verbose',0);
%xTrn=Xtrn.';
net = trainNetwork(Xtrn,yTrn,layers,options);