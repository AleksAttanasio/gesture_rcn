outputLine = [  'Current training network: ' newline ...
                'Samples: ' num2str(size(t,2)) newline ...
                'Neurons: ' num2str(neuronsNumber(neuronIndex)) newline ...
                'Regulation: ' num2str(regValues(regIndex)) newline ...
                'Data division: ' num2str(datasetDivision(dataDivIndex, :))];
disp(outputLine)