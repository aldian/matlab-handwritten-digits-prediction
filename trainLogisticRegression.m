% Train logistic regression
% Before running this script, you need to be sure file mnist_data.mat
% exists in the current directory.
% If it doesn't exist, please run mnistUrlToMatFile first.
clear; close all; clc;

num_labels = 10;

% Load data from MNIST files
load mnist_data.mat;

lambda = 0.1;
[all_theta] = oneVsAll(XTrain, yTrain, num_labels, lambda); % Train logistic regression
save lr_theta.mat all_theta;