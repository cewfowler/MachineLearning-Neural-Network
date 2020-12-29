function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% sigmoid function
hx = sigmoid(X * theta);

% Compute cost for logistic regression
J = (1/m) * ((-y)' * log(hx) - (1 - y)' * log(1 - hx));

% Add regularization params
J = J + sum((lambda/(2*m)) * theta(2:size(theta),:).^2);

% Calculate gradient for logistic regression
grad = (1/m) * X' * (hx - y);

% Add regularization params
grad(2:end) = grad(2:end) + (lambda/m) * theta(2:end);

end
