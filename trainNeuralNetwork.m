% trainNeuralNetwork.m
% Before running this script, you need to be sure file mnist_data.mat
% exists in the current directory.
% If it doesn't exist, please run mnistUrlToMatFile first.
clear; close all; clc;

num_labels = 10;

% Load data from MNIST files
load mnist_data.mat;

input_layer_size = 28 * 28;
hidden_layer_size = 50;

% Train the neural network
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:); initial_Theta2(:)];
options = optimset('MaxIter', 60);
lambda = 1;
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, XTrain, yTrain, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

save nn_theta.mat Theta1 Theta2