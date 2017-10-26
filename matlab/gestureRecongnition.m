%% Pattern Recognition for Gesture
clc 
clear all
close all
%% Loading dataset, initializing Network and running learning

% loading dataset
[x,t] = gesture_dataset('dataset_109320_onehot_noGPU');

% initializing network with numbers of neuron
net = patternnet(400);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 10/100;

% view(net) % ++++++++ uncomment to view network scheme ++++++++

% training network
[net,tr] = train(net,x,t); 
nntraintool


%% Test the network

% load test data
testX = x(:,tr.testInd);
testT = t(:,tr.testInd);

% test network and convert indices in num
testY = net(testX);
testIndices = vec2ind(testY);

saveNet

%% Plots and print data
% figure
% plotperform(tr) % Plot performance (MSE)
figure
plotconfusion(testT,testY) % Plot confusion matrix
% figure
% plotroc(testT,testY) % Plot ROC (receiver operating characteristic)

[c,cm] = confusion(testT,testY);

fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

%% Test manually the network
load dataset_test.mat % Loading dataset
load softmaxNet_400x109320_99_0.mat

% Manually enter new samples (knot_tying)
testSet = knot_tying; % ++++ CHANGE [] to variables to set +++++ dimension are [features x timeStep]
res = net(testSet);
res = vec2ind(res);

err = knot_tyingLabels - res;
idx = err==0;
accuracy = sum(idx(:))/numel(res);

%%
% Manually enter new samples (needle_passing)
load dataset_test.mat % Loading dataset
load softmaxNet_400x109320_99_0.mat

testSet = needle_passing; % ++++ CHANGE [] to variables to set +++++ dimension are [features x timeStep]
res = net(testSet);
res = vec2ind(res);

err = needle_passingLabels - res;
idx = err==0;
accuracy = sum(idx(:))/numel(res);

