function [inputs,targets] = gesture_dataset (name)
%Gesture dataset

load(name)
inputs = Training_set';
targets = labels';
