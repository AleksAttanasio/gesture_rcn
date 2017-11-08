%% Auto Train different neural networks and save results in a table
clc
clear all 
close all

%% Loop for automatic training

% Load dataset for testing and table with results
load test_auto; % dataset for testing in two cells with labelSet and testSet (1 x trial_num)
load results_table;
% dev = gpuDevice(1); % Uncomment to use GPU device (not stable)

% Dataset name (labels and training set must be: <samples> x <features>)
datasetName = 'dataset_246783_onehot_noGPU';

% Set of number to create networks with different number of neurons
neuronsNumber = [280 330 380 430 480 530 580 650];

% Dataset division for training, validation and test (default =
% 70%/20%/10%). It is possible to set different dataset division
% parameters by adding them in a new row.
datasetDivision = [[70/100 20/100 10/100]]; 

% Set of different numbers for regularization. Regularization helps the
% network not to get too close to the minimum error in order to guarantee a
% better generalization over results.
regValues = [0 0.2 0.4 0.6 0.8];  

% for loops to train several networks (change the order of loops to change
% order of training).
for neuronIndex = 1:numel(neuronsNumber)
    for dataDivIndex = 1:size(datasetDivision,1)
        for regIndex = 1:numel(regValues)
        
            % Load dataset with function "gesture_dataset"
            [x,t] = gesture_dataset(datasetName);

            % initializing network with numbers of neuron
            net = patternnet(neuronsNumber(neuronIndex));
            
            % Datset division
            net.divideParam.trainRatio = datasetDivision(dataDivIndex, 1);
            net.divideParam.valRatio = datasetDivision(dataDivIndex, 2);
            net.divideParam.testRatio = datasetDivision(dataDivIndex, 3);
            
            % Set regularization 
            net.performParam.regularization = regValues(regIndex);
            % net.output.processFcns = {'mapminmax'};

            % view(net) % Uncomment to see the network scheme
            
            printCurrentNetwork;
            
            % training network with Neural Network Toolbox
            [net,tr] = train(net,x,t); 
            nntraintool
            
            % load test data
            testX = x(:,tr.testInd);
            testT = t(:,tr.testInd);

            % test network and convert inferences as an integer
            testY = net(testX);
            testIndices = vec2ind(testY);

            % Evaluate accuracy of network
            [c,cm] = confusion(testT,testY);
            netAccuracy = 100*(1-c);
            performance = tr.best_vperf;
            
            % save Network with "saveNet.m"
            saveNet
            
            % Store test accuracy for every sequence
            testAccuracy = zeros(1,numel(testSet));
            for testIndex = 1:numel(testSet)
                
                % Manually enter new samples (knot_tying)
                test = testSet{1,testIndex}; 
                res = net(test);
                res = vec2ind(res);

                err = labelSet{1,testIndex} - res;
                idx = err==0;
                accuracy = sum(idx(:))/numel(res);
                testAccuracy(testIndex) = accuracy;
            end
            
            % Populate new row in table
            newDataRow = {name, neuronsNumber(neuronIndex), size(t,2), ...
                            regValues(regIndex), datasetDivision(dataDivIndex, :), ...
                            netAccuracy, performance, testAccuracy, ...
                            mean(testAccuracy), std(testAccuracy)};
            
            % Add results to table
            resTable = [resTable; newDataRow];
            
            % Save updated results table
            save('results_table.mat','resTable');
        end
    end
end


