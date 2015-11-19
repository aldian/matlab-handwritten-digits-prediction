function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); 

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); 

m = size(X, 1);
J = 0;

Theta1T = Theta1';
Theta2T = Theta2';
z2 = [ones(m, 1) X] * Theta1T;
a2 = sigmoid(z2);
z3 = [ones(m, 1) a2] * Theta2T;
a3 = sigmoid(z3);
theta1_rest = Theta1(:, 2:end);
theta2_rest = Theta2(:, 2:end);

for c = 1:num_labels
    yc = y == (c - 1);
    yct = yc';
    h_of_X = a3(:, c);
    J = J + (-yct * log(h_of_X) - (1 - yct) * log(1 - h_of_X)) / m;
end

if lambda > 0    
    reg = 0;
    for i = 1:size(theta1_rest, 2)
        th = theta1_rest(:, i);
        reg = reg + th' * th;
    end

    for i = 1:size(theta2_rest, 2)
        th = theta2_rest(:, i);
        reg = reg + th' * th;
    end
    J = J + lambda * reg/2/m;
end

yx = zeros(m, num_labels);
for i = 1:m
    label = y(i);
    yx(i, label+1) = 1;
end
d3 = a3 - yx;
d2 = d3 * theta2_rest .* sigmoidGradient(z2);  
Dt1 = d2' * [ones(m, 1) X];
Dt2 = d3' * [ones(m, 1) a2];
Theta1_grad = (Dt1 + lambda * [zeros(size(Theta1, 1), 1) theta1_rest]) / m;
Theta2_grad = (Dt2 + lambda * [zeros(size(Theta2, 1), 1) theta2_rest]) / m;

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
