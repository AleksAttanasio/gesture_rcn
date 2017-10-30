%% Auto Train different neural networks and save results in a table
clc
clear all 
close all

%% Loop for automatic training

load test_auto;
load results_table;

datasetName = 'dataset_246783_onehot_noGPU';
neuronsNumber = [130 180 230 280 330 380 430 480];
datasetDivision = [[70/100 20/100 10/100]; [80/100 15/100 5/100]];
regValues = [0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];


for neuronIndex = 1:numel(neuronsNumber)
    for dataDivIndex = 1:size(datasetDivision,1)
        for regIndex = 1:numel(regValues)
        
            [x,t] = gesture_dataset(datasetName);

            % initializing network with numbers of neuron
            net = patternnet(neuronsNumber(neuronIndex));
            net.divideParam.trainRatio = datasetDivision(dataDivIndex, 1);
            net.divideParam.valRatio = datasetDivision(dataDivIndex, 2);
            net.divideParam.testRatio = datasetDivision(dataDivIndex, 3);
            net.performParam.regularization = regValues(regIndex);

            % view(net) % ++++++++ uncomment to view network scheme ++++++++
            
            printCurrentNetwork;
            
            % training network
            [net,tr] = train(net,x,t); 
            nntraintool
            
            % load test data
            testX = x(:,tr.testInd);
            testT = t(:,tr.testInd);

            % test network and convert indices in num
            testY = net(testX);
            testIndices = vec2ind(testY);

            % Evaluate accuracy of network
            [c,cm] = confusion(testT,testY);
            netAccuracy = 100*(1-c);
            performance = tr.best_vperf;
            
            saveNet
            
            % Store test accuracy for every sequence
            testAccuracy = zeros(1,numel(testSet));
            for testIndex = 1:numel(testSet)
                
                % Manually enter new samples (knot_tying)
                test = testSet{1,testIndex}; % ++++ CHANGE [] to variables to set +++++ dimension are [features x timeStep]
                res = net(test);
                res = vec2ind(res);

                err = labelSet{1,testIndex} - res;
                idx = err==0;
                accuracy = sum(idx(:))/numel(res);
                testAccuracy(testIndex) = accuracy;
            end
            
            newDataRow = {name, neuronsNumber(neuronIndex), size(t,2), ...
                            regValues(regIndex), datasetDivision(dataDivIndex, :), ...
                            netAccuracy, performance, testAccuracy, mean(testAccuracy)};
            resTable = [resTable; newDataRow];
            
            save('results_table.mat','resTable');
        end
    end
end


