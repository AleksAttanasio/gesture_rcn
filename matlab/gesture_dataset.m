function [inputs,targets] = gesture_dataset (name)
%Gesture dataset

load(name)
inputs = training';
targets = labels';
