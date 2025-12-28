%Omar Morsy
%oam@pitt.edu
%Homework 2 
%ECE 1395 - Machine Learning


%Question1- The Cost function

disp('Question 1: Testing computeCost function');   %Displaying that line

% X contains x1 and x2 (features).  y is the actual output
X = [0 1; 1 1.5; 2 4; 3 2];
y = [1.5; 4; 8.5; 8.5];

% Adding bias term (x0 = 1) column to the matrix X 
X = [ones(size(X,1),1) X];

% Testing three thetas
theta1 = [0.5; 2; 1];
theta2 = [10; -1; -2];
theta3 = [0.5; 1; 2];

%Computing the cost for each case by calling computeCost function
cost_1 = computeCost(X, y, theta1);
cost_2 = computeCost(X, y, theta2);
cost_3 = computeCost(X, y, theta3);

%Using fprintf to be able to display with rounding 4 decimal numbers
fprintf('Cost for theta1 = [0.5 2 1]: %.4f\n', cost_1); 
fprintf('Cost for theta2 = [10 -1 -2]: %.4f\n', cost_2);
fprintf('Cost for theta3 = [0.5 1 2]: %.4f\n', cost_3);

%I got the same answers when I computed it by manually by my self 0.0000,
%18.5938 and 0.7812




%Question2- Gradient descent

disp('Question 2: Testing gradientDescent function');  %Displaying the header for question 2

%Calling the gradientDescent function with learning rate equal to 0.001 and
%15 iterations
[theta_grad, cost_history] = gradientDescent(X, y, 0.001, 15);

%Using fprintf to be able to display and round 4 decimal numbers
fprintf('Theta from gradient descent: [%.4f %.4f %.4f]\n', theta_grad);  
fprintf('Final cost: %.4f\n', cost_history(end)); %I used end to get the last value in the vector cost_history which is after 15 iterations

%Theta from gradient descent: [0.7740 0.0498 0.8431], Final cost: 7.0573


% Question 3: Testing the normalEqn function
disp('Question 3: Testing normalEqn function'); %Displaying the header line for question 3

%Calling the normalEqn function to compute theta_normal (theta from normal
%equation)
theta_normal = normalEqn(X, y);
fprintf('Theta from normal equation: [%.4f %.4f %.4f]\n', theta_normal); %Displaying theta from normal equation


% Question 4: Linear regression with one variable
disp('Question 4: Car price prediction'); %Displaying the header line for question 4

%PART A: Loading the data
data = readmatrix('hw2_data1.csv');
X = data(:,1);  %The Horse power
y = data(:,2);  %The Price

%PART B: Plotting data
figure;  %Building a figure
scatter(X, y, 'x'); %scattering with x markers
xlabel('Horse Power'); %The title of axis x
ylabel('Price'); %The title of axis y
title('Car Prices vs Horse Power');  %The title of the plot
saveas(gcf, 'ps2-4-b.png');  %Saving the image

%PART C: Adding bias term
X = [ones(size(X,1),1) X];   %Model becomes: h(x) = θ₀ + θ₁x₁

% Displaying the sizes
fprintf('Size of feature matrix X: %d x %d\n', size(X,1), size(X,2));
fprintf('Size of label vector y: %d x %d\n', size(y,1), size(y,2));

%Part D: Splitting data
m = size(X,1); % m = 178 (total examples)
train_idx = randperm(m, round(0.9*m)); % Calculating how many examples for training (90%).   % round(0.9 * 178) = 160
test_idx = setdiff(1:m, train_idx); % Generating test indices (all indices not in train_idx)
% Creating training and test sets
X_train = X(train_idx,:);
y_train = y(train_idx);
X_test = X(test_idx,:);
y_test = y(test_idx);

%Part e Training model using gradient descent
[theta, cost_history] = gradientDescent(X_train, y_train, 0.3, 500);  %calling the gradient descent function

% Plotting the cost history
figure; %Creating new figure 
plot(1:500, cost_history);  %Plotting
xlabel('Iteration');  %The title of x axis
ylabel('Cost'); %The title of y axis
title('Cost vs Iteration'); %The title of the plotting
saveas(gcf, 'ps2-4-e.png'); %Saving the image

%PART F:  Plotting the fitted line
figure;  %New window 
scatter(X_train(:,2), y_train, 'x');  %scattering with x markers
hold on; % Keeping the plot for adding the line
x_plot = [min(X(:,2)); max(X(:,2))]; %using two points the max and min of the horse power
y_plot = [ones(2,1) x_plot] * theta; % Calculating corresponding y values using our model. Calculating y = θ₀ + θ₁x for each x value
plot(x_plot, y_plot, '--'); % Plotting with dashed line style
xlabel('Horse Power');  %The title of xaxis
ylabel('Price'); %The title of y axis
title('Fitted Line with Training Data'); %The title of the plot
legend('Training Data', 'Fitted Line'); %Using legend
saveas(gcf, 'ps2-4-f.png'); %Saving the image of the plot

%PART G: Computing the test error
y_pred = X_test * theta; %prediction y
test_error = mean((y_pred - y_test).^2);
fprintf('Test error (gradient descent): %.4f\n', test_error);  %Displaying the test error   12.4083

%PART H:  Normal equation solution
theta_normal = normalEqn(X_train, y_train);   %Calling the normal equation function
y_pred_normal = X_test * theta_normal;   %Calculating y-prediction
test_error_normal = mean((y_pred_normal - y_test).^2); %Calculating the error  12.4074
fprintf('Test error (normal equation): %.4f\n', test_error_normal); %displaying

%PART I: Learning rate study
alphas = [0.001 0.003 0.03 3];   %alphas values
for i = 1:length(alphas) %Starting for loop until the length of the vector alphas which is 4 (four iterations)
    [~, cost_history] = gradientDescent(X_train, y_train, alphas(i), 300);   %Calling the gradient descent function
    figure;  %New window
    plot(1:300, cost_history);  %plotting
    xlabel('Iteration');  %The title of axis x
    ylabel('Cost');  %The title of axis y
    title(sprintf('Cost vs Iteration (alpha=%.3f)', alphas(i)));  %The title of the plot
     grid on;
    saveas(gcf, sprintf('ps2-4-i-%d.png', i)); %saving the image
end  %The end for the for loop



% Question 5: Linear regression with multiple variables
disp('Question 5: CO2 emission prediction'); %displaying the the header line for question 5

%PART A:
% Loading data from the second file
data = readmatrix('hw2_data3.csv'); 
X = data(:,1:2);  % Engine size and weight
y = data(:,3);    % CO2 emission

% Standardizing features
X_mean = mean(X);  %getting the mean
X_std = std(X);  %the standard deviation for each feature dimension
X_norm = (X - X_mean) ./ X_std;

% Adding bias term
X_norm = [ones(size(X_norm,1),1) X_norm];

disp(X_mean);
disp(X_std);

%PART B: Training model
[theta, cost_history] = gradientDescent(X_norm, y, 0.01, 750);  %Calling function gradient descent 

% Plotting cost history
figure; %New window for the figure 
plot(1:750, cost_history);   %Plotting
xlabel('Iteration');  %The title of x axis
ylabel('Cost'); %The title of y axis
title('Cost vs Iteration (Multiple Variables)'); %The title of the plot 
saveas(gcf, 'ps2-5-b.png');  %Saving the image

% Making prediction for new car
new_features = [2100 1200];   %vector of new features
new_features_norm = (new_features - X_mean) ./ X_std; 
new_features_norm = [1 new_features_norm];
prediction = new_features_norm * theta;  %The predictions

fprintf('Predicted CO2 emission: %.2f\n', prediction); %Displaying the prediction