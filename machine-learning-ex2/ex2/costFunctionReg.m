function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% fprintf('\nSize theta %f\n', size(theta));


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



z = X*theta;
h = sigmoid(z);
regular = ones(size(theta), 1) * (lambda / (2*m));
regular(1, 1) = 0;
J = m^-1 * (-y'*log(h) - (1 - y')*log(1 - h)) + regular' * theta.^2


regular_2 = ones(size(theta), 1) * (lambda / m);
regular_2(1, 1) = 0;
term_2 = theta.* regular_2;

grad = m^-1 * (h' - y')*X + term_2';



% =============================================================

end
