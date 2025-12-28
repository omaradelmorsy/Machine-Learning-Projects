function [theta, cost_history] = gradientDescent(X_train, y_train, alpha, iters)   % This function is to compute the Gradient descent for linear regression
  
    % X_train is matrix  m x (n+1)  where m is the number of examples and n is the number of features
    % y_train is vector m x 1 of  the actual (target) values
    % alpha is the learning rate.   iters is the number of iterations.
   %cost_history is the cost at each iteration.
    
    % Getting dimensions
    m = length(y_train);  %using length to find the number of examples.
    n = size(X_train, 2);  % Number of features (including bias term)
    
    % Initializing theta randomly using randn function
    theta = randn(n, 1);
    
    % Initializing the cost history. This is my starting point. Creating vector to store cost at each iterationSize is iters×1, initialized with zeros
    cost_history = zeros(iters, 1);
    
    % Gradient descent   using for loop for the iterations
    for iter = 1:iters
        % Computing predictions
        predictions = X_train * theta;
        
        % Computing gradients
        %Calculating gradients using the formula:
%predictions - y_train: Error for each example (m×1)
%X_train': Transpose of X_train (n×m)
%X_train' * (predictions - y_train): Sum of errors weighted by features (n×1)
%(1/m): Average over all examples
%This is ∂J/∂θ = (1/m)X'(Xθ - y)
        gradients = (1/m) * X_train' * (predictions - y_train);
        
        % Updating parameters by using gradient decsent rule: θ = θ - α∇J(θ)
        theta = theta - alpha * gradients;
        
        % Storing the cost
        cost_history(iter) = computeCost(X_train, y_train, theta);  %This will help to monitor convergence
    end  %the end of the for loop 
end  %The end of the function