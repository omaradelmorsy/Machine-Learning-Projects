function [y_predict] = logReg_multi(X_train, y_train, X_test)
% logReg_multi Implements one-vs-all approach using logistic regression
% inputs: X_train is an mx(n+1) feature matrix with m samples and n feature dimensions
% y_train is an mx1 vector containing the labels for training instances
% X_test is an dx(n+1) feature matrix with d test instances
% output: y_predict is a dx1 vector containing predicted labels for test instances

% Getting number of classes from training data
classes = unique(y_train);  %It gets unique class labels from training data
num_classes = length(classes); % It counts number of unique classes
num_test = size(X_test, 1); %It gets number of test examples

% Initializing matrix to store probabilities for each class
probabilities = zeros(num_test, num_classes); % It creates matrix to store probabilities

% Training one classifier for each class
for i = 1:num_classes
    % Creating binary labels for current class (1 for current class, 0 for others)
    binary_labels = double(y_train == classes(i));
    
    % Training logistic regression model for current class
    mdl = fitclinear(X_train, binary_labels, 'Learner', 'logistic');
    
    % Getting probability predictions for positive class
    [~, probs] = predict(mdl, X_test);
    probabilities(:, i) = probs(:, 2);  % Storing probability of positive class
end

% For each test instance, choosing class with highest probability
[~, max_idx] = max(probabilities, [], 2);
y_predict = classes(max_idx);

end