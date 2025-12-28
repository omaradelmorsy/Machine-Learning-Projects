function [theta] = reg_normalEqn(X_train, y_train, lambda)
% Reg_normalEqn Computes closed-form solution to linear regression using normal equation
% with regularization
% inputs: X_train is an mx(n+1) feature matrix with m samples and n feature dimensions. m is the 
% number of the samples in the training set.
% y_train is an mx1 vector containing the output for the training set. The i-th element
% in y_train should correspond to the i-th row (training sample) in X_train
% lambda, the value of the regularization parameter λ.
% outputs: theta is a (n+1)x1 vector of weights (one per feature dimension).

% Getting the size of X_train
[~, n] = size(X_train);

% Creating the regularization matrix 
reg_matrix = eye(n) * lambda;  %eye(n) is identity matrix n*n
reg_matrix(1,1) = 0; % I will not regularize bias term

% Calculating theta using normal equation with regularization
theta = pinv(X_train' * X_train + reg_matrix) * X_train' * y_train;

end