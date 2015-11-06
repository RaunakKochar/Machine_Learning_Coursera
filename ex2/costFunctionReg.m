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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


hyp = X*theta;
sig_hyp = sigmoid(hyp);
cost = - (y.*log(sig_hyp) + (1 - y).*log(1 - sig_hyp));
J = (1/m) * sum(cost) + lambda / (2*m) * (sum(theta.*theta) - theta(1)*theta(1));

grad(1) = (1/m)* sum((sig_hyp - y).*X(:,1));

for i = 2:size(theta),
  grad(i) = (1/m)* sum((sig_hyp - y).*X(:,i)) + lambda * theta(i) / m;
end




% =============================================================

end
