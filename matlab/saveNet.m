%% Saving Network and training history
% evaluating accuracy
[c,cm] = confusion(testT,testY);
acc = 100*(1-c);

% declaring index
k = 0;
% name format: 'softmaxNet_<netSize>x<trainSamples>_<accuracy>_<netIndex>'
name = strcat('network_models/softmaxNet_',num2str(net.layers{1,1}.size),'x',num2str(size(t,2)),'_',num2str(round(acc)),...
        '_',num2str(k));

% check for existing file
while (exist(strcat(name,'.mat')) == 2)
    
    k = k+1; % increment index
    name(end) = []; % remove index
    name = strcat(name,num2str(k)) % change index
    
end

%Check for regulation parameter
if(net.performParam.regularization ~= 0)
    
    % convert regulation parameter to string
    reg = replace(num2str(net.performParam.regularization),'.','');
    reg = reg(1:2); % save just significant decimals
    name = strcat(name,'_reg',reg);
    
end

% save network
save (name, 'net', 'tr')