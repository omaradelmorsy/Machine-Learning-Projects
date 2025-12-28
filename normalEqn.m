function [theta] = normalEqn(X_train, y_train)
    % Normal equation solution for linear regression
    % X_train is matrix m x (n+1) where m is the number of examples and n is the number of features
    % y_train is vector m x 1 of the (actual) target values
    
    % Computing theta using normal equation. θ = (X'X)⁻¹X'y
    theta = pinv(X_train' * X_train) * X_train' * y_train;
end