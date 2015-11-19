% predictTestDataUsingNN.mat
% Before running this script, you need to be sure file nn_theta.mat
% exists in the current directory.
% If it doesn't exist, please run trainNeuralNetwork first.
clear; close all; clc;

load mnist_data.mat
load nn_theta.mat

pred = predict(Theta1, Theta2, XTrain); % Predict the training set
fprintf('\nNeural Network Training Set Accuracy: %f%%\n', mean(double(pred == yTrain)) * 100);
pred = predict(Theta1, Theta2, XTest); % Predict the test set
fprintf('\nNeural Network Test Set Accuracy: %f%%\n', mean(double(pred == yTest)) * 100);