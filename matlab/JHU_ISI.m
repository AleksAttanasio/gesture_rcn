%% OPEN JHU_ISI dataset and MERGE with gestures

clear 
close all
clc

%% 

% 0) Load GPU device, uncomment this line to enable GPU device reading
% dev = gpuDevice(1); 

% 1) Loading
Data_withGestures = Loader_Gestures();

% 2) Features scaling and mean normalization [5] Grouping [6]
Data_withGestures = Scaling_Grouping(Data_withGestures);

% 3) Removing of the labels
Training_set=Data_withGestures(:,3:end);

% 4) Gathering labels from Training Set
labels = zeros(size(Training_set,1),15);

for k = 1:1:size(Data_withGestures,1)
    labels(k,Data_withGestures(k,end)) = 1;
end

Training_set(:,end) = [];

%% NARX System setting up and training
% Solve an Autoregression Problem with External Input with a NARX Neural Network

X = tonndata(Training_set,false,false);
T = tonndata(labels,false,false);

trainFcn = 'trainscg';  % Scaled Conjugate Gradient.

% Create a Nonlinear Autoregressive Network with External Input
inputDelays = 1:50;
feedbackDelays = 1:30;
hiddenLayerSize = 120;
net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);
net.output.processFcns = {'mapminmax'};
net.trainParam.epochs = 6000;

% Prepare the Data for Training and Simulation
[x,xi,ai,t] = preparets(net,X,{},T);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 75/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 10/100;

% Train the Network
[net,tr] = train(net,x,t,xi,ai); % Add 'useGPU','yes' to enable GPU computing

% Test the Network
y = net(x,xi,ai);
e = gsubtract(t,y);
performance = perform(net,t,y)

saveNet;

% View the Network
view(net)

%% Plot results graph

% Plots
figure, plotperform(tr)
figure, plottrainstate(tr)
figure, ploterrhist(e)
figure, plotregression(t,y)
figure, plotresponse(t,y)
figure, ploterrcorr(e)
figure, plotinerrcorr(x,e)


