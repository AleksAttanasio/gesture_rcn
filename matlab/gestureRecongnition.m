%% Pattern Recognition for Gesture
clc 
clear all
close all
%% Loading dataset, initializing Network and running learning

% loading dataset
[x,t] = gesture_dataset('dataset_109320_onehot_noGPU');

% initializing network with numbers of neuron
net = patternnet(120); 

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
%% Test manually the network
load dataset_test.mat % Loading dataset
load softmaxNet_120x109320_98_0.mat


% Manually enter new samples 
testSet = knot_tying; % ++++ CHANGE [] to variables to set +++++
res = net(testSet);
res = vec2ind(res);

%% Plots and print data
figure
plotperform(tr) % Plot performance (MSE)
figure
plotconfusion(testT,testY) % Plot confusion matrix
figure
plotroc(testT,testY) % Plot ROC (receiver operating characteristic)

[c,cm] = confusion(testT,testY);

fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);