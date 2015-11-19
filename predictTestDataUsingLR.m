% Predit Test Data using LR
% Before running this script, you need to be sure file lr_theta.mat
% exists in the current directory.
% If it doesn't exist, please run trainLogisticRegression first.
clear; close all; clc;

load mnist_data.mat
load lr_theta.mat

pred = predictOneVsAll(all_theta, XTrain); % Predict the training set
fprintf('\nLogistic Regression Training Set Accuracy: %f%%\n', mean(double(pred == yTrain)) * 100);
pred = predictOneVsAll(all_theta, XTest); % Predict the test set
fprintf('\nLogistic Regression Test Set Accuracy: %f%%\n', mean(double(pred == yTest)) * 100);