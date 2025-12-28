%Omar Morsy
%oam15@pitt.edu
%ECE 1395   HomeWork 4



% 1. Regularization
fprintf('\n1. Regularization\n');  %Printing the header
%Part a: The function Reg_normalEqn has been implemented to compute the
%closed-form solution to linear regression using normal equation with
%regularization

%Part b: 
% Loading the data
fprintf('\nLoading hw4_data1.mat...\n'); %Printing the header line 
load('hw4_data1.mat');  %Loading the data from the hw4_data1.mat

% Adding the bias term (column of ones) to X_data
X_data = [ones(size(X_data,1),1) X_data];
%X_data: [1001×500 double]
        % y: [1001×1 double]    The size of feature matrix: 1001 x 500

%Part c:
% Defining the lambda values
lambda_values = [0 0.001 0.003 0.005 0.007 0.009 0.012 0.017];
num_iterations = 20;  %Definning the number of the iteration

% Initializing the matrices to store errors
train_errors = zeros(num_iterations, length(lambda_values));  %train errors (matrix)
test_errors = zeros(num_iterations, length(lambda_values));  %test errors  (matrix)

% Loop for cross-validation
for iter = 1:num_iterations
    % Randomly split data (85% training, 15% testing)
    n = size(X_data, 1);
    train_idx = randperm(n, round(0.85*n));  %Randomly selecting 85% for training
    test_idx = setdiff(1:n, train_idx);   %the remaining 15% for testing
    
    X_train = X_data(train_idx, :);  %Extracting x_train
    y_train = y(train_idx);  %Extracting y train
    X_test = X_data(test_idx, :);   %Extracting x test 
    y_test = y(test_idx);  %Extracting y test 
    
    % Iterating over lambda values
    for l = 1:length(lambda_values)
        lambda = lambda_values(l);
        
        % Training the model using Normal Equation
        theta = reg_normalEqn(X_train, y_train, lambda);
        
        % Computing the errors
        train_errors(iter, l) = mean((X_train * theta - y_train).^2) / 2;  %the train errors
        test_errors(iter, l) = mean((X_test * theta - y_test).^2) / 2;  %the test errors
    end
end  

% Computing the average errors
avg_train_error = mean(train_errors);  %the average train error
avg_test_error = mean(test_errors);  %the average test error

% Plotting results
figure;  %New window
plot(lambda_values, avg_train_error, '-r*', 'LineWidth', 1.5, 'DisplayName', 'Training Error');  %Plotting the data
hold on;
plot(lambda_values, avg_test_error, '-bo', 'LineWidth', 1.5, 'DisplayName', 'Testing Error');  %Plotting
xlabel('\lambda');  %The title of x axis 
ylabel('Average Error'); %The title of y axis 
title('Average Training and Testing Error vs \lambda');  %The title of the plot
legend('show', 'Location', 'best');  %Using legend
grid on;  %Adding grid lines
saveas(gcf, 'ps4-1-a.png');  %saving the plot image 
%Based on the graph of training and testing errors versus λ, I suggest using λ=0.003-0.005 as it achieves the minimum testing error while maintaining a reasonable gap with training error. 
% Without regularization (λ=0), the model overfits as shown by high testing error, while larger λ values lead to underfitting with both errors increasing. The chosen λ value provides the optimal balance between model complexity and generalization ability, 
% as evidenced by the convergence of training and testing errors at this point.


% 2. KNN - Effect of K
fprintf('\n2. KNN - Effect of K\n');  %Displaying the header line for question 2
load('hw4_data2.mat');  %Loading the data
K_values = 1:2:15;  % k values
num_folds = 5;
accuracies = zeros(length(K_values), num_folds);  %accuracies matrix

% Creating cell arrays to store the 5 folds of data
X_folds = {X1, X2, X3, X4, X5}; % Storing feature matrices for each fold
y_folds = {y1, y2, y3, y4, y5}; % Store corresponding labels for each fold
% Outer loop: iterating through different K values
for k_idx = 1:length(K_values)
    k = K_values(k_idx); % Getting current K value for KNN
     % Inner loop: performing 5-fold cross validation
    for fold = 1:num_folds
        % Creating training and testing sets
         % Selecting current fold as test set
        test_idx = fold;  % Current fold becomes test set
        train_idx = setdiff(1:num_folds, fold); % Remaining folds become training set
       
        % Extracting test data for current fold
        X_test = X_folds{test_idx};
        y_test = y_folds{test_idx};
        % Combining remaining folds for training data
        X_train = vertcat(X_folds{train_idx});  % Vertically concatenate training features
        y_train = vertcat(y_folds{train_idx}); % Vertically concatenate training labels
        
        % Converting y_train to numeric or categorical if needed
        if iscell(y_train)   % If labels are cell array
            y_train = categorical(y_train); % Converting to categorical
        elseif ~isnumeric(y_train)  % If labels are not numeric
            y_train = double(y_train); % Converting to double
        end
        
        % Ensuring X_train and y_train are non-empty
        if isempty(X_train) || isempty(y_train)
            warning('Skipping fold %d: Empty training data.', fold);  % Skipping to next iteration if data is empty
            continue;
        end
        
        % Ensuring y_train has at least two unique labels
        if numel(unique(y_train)) < 2
            warning('Skipping fold %d: Only one unique label found.', fold);   % Skipping to next iteration if only one class present
            continue;
        end
        
        %ATTENTION!!!!! PLEASEEEEEE TO RUN THIS PART YOU NEED TO USE MATLAB
        %online
        %OR Statistics and Machine Learning Toolbox BECAUSE OF THIS
        %FUNCTION (fitcknn)
         
        % Training KNN model
        mdl = fitcknn(X_train, y_train, 'NumNeighbors', k);
        y_pred = predict(mdl, X_test);  %y prediction
        accuracies(k_idx, fold) = mean(y_pred == y_test);  %accuracies
    end
end

avg_accuracies = mean(accuracies, 2);  %the average accuracies
figure;  %new window
plot(K_values, avg_accuracies, '-o');  %Plotting
xlabel('K');  %The title of axis x
ylabel('Average Accuracy');  %The title of axis y
title('KNN Classification: Accuracy vs K');  %The title of the plot figure
grid on;  %Adding grid lines 
saveas(gcf, 'ps4-2-a.png');  %Saving the plot image

%K=9 achieves the highest accuracy of 0.966 and is the recommended value for this specific dataset. 
% However, this K value is not necessarily robust for other classification problems since the optimal K depends heavily on each dataset's unique characteristics like size, 
% dimensionality, and class distribution. For any new problem, cross-validation should be performed to find the optimal K value, as what works well for one dataset may not be optimal for another

% 3. One-vs-All Classification
fprintf('\n3. One-vs-All Classification\n');  %Printing the question 3 header
load('hw4_data3.mat');  %loading the data from the file hw4_data3.mat
y_pred_train = logReg_multi(X_train, y_train, X_train); %calling the function logReg_multi to get y_predict train  
y_pred_test = logReg_multi(X_train, y_train, X_test); %calling the function logReg_multi to get y_predict test 
train_accuracy = mean(y_pred_train == y_train);  %Computing train accuracy
test_accuracy = mean(y_pred_test == y_test);  %Computing test accuracy 
fprintf('\nTraining Accuracy: %.4f\n', train_accuracy);  %Displaying the train accuracy 
fprintf('Testing Accuracy: %.4f\n', test_accuracy); %Displaying the test accuracy

