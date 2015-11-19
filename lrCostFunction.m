function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

h_of_X = sigmoid(X * theta); % (m * (n + 1) matrix) * ((n + 1) * 1 matrix) resulting m * 1 matrix
J = (-y' * log(h_of_X) - (1 - y') * log(1 - h_of_X)) / m;
grad = (X' * (h_of_X - y)) ./ m;

theta_rest = theta(2:end, :);

% calculate regularization
J = J + lambda * (theta_rest' * theta_rest) / 2 / m;
grad = grad + [0; lambda .* theta_rest ./ m];

grad = grad(:);

end
