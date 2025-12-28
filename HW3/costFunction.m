function [J, grad] = costFunction(theta, X, y)
%The cost function will Compute the cost and the gradient for the logistic regression
%   [J, grad] = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   Note: grad is optional and only computed if requested

if nargin < 3
    error('Not enough input arguments. Need theta, X, and y.');   %Getting an error if the argument not enough
end

m = length(y); % The number of the training examples

% Computing the hypothesis  with call the sigmoid function
h = sigmoid(X * theta);

% Computing the cost
J = (1/m) * sum(-y .* log(h) - (1-y) .* log(1-h));

% Computing the gradient if requested
if nargout > 1
    grad = zeros(size(theta));
    for j = 1:length(theta)  
        grad(j) = (1/m) * sum((h - y) .* X(:,j));  %Computing the gradient
    end
end

end   %The end of the function