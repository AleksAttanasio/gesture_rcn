%% Load and organize Dataset
clc
close all
clear all

%%

% 0) Load GPU device, uncomment this line to enable GPU device reading
% dev = gpuDevice(1); 

% 1) Loading
Data_withGestures = Loader_Gestures();

% 1.1) Index for Grouping the code will generate numel(k) datasets
k = [1];

for i = 1:1:numel(k)
    % 2) Features scaling and mean normalization [5] Grouping [6]
    Data_withGestures = Scaling_Grouping(Data_withGestures, k(i));
    [r,~]=find(isnan(Data_withGestures));
    Data_withGestures(r,:)=[];
    
    % 3) Removing of the labels
    training=Data_withGestures(:,3:end);

    % 4) Gathering labels from Training Set
    labels = zeros(size(training,1),15);

    for j = 1:1:size(Data_withGestures,1)
        labels(j,Data_withGestures(j,end)) = 1;
    end

    % Remove label column
    training(:,end) = [];
    
    % 5) Defining name for dataset
    name = strcat('dataset_',num2str(size(training,1)),'_onehot');
    
    % Check for GPU instance
    if (exist('dev','var') == 0)       
        name = strcat(name, '_noGPU');     
    end
    
    save(name, 'training', 'labels')
end