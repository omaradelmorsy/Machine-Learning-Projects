function J = computeCost(X, y, theta)
    % I am computing the cost for linear regression
    % X is matrix with  m x (n+1) where m is the number of examples and n is the number of features
    % y is vector m x 1  of  the target values
    % theta is vector (n+1) x 1  of  the parameters
    
    m = length(y); % I used the built-in function (length) to get the number of training examples
    
    % Computing predictions (The estimated value for Y)   
    predictions = X * theta; 
    
    % Computing squared errors
    squared_Errors = (predictions - y).^2;
    
    % Computing the cost
    J = (1/(2*m)) * sum(squared_Errors);   %Using the sum function to determine the summation.
end