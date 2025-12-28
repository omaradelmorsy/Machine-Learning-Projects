%Omar Morsy
%oam15@pitt.edu
%Homework 3 
%ECE 1395 - Machine Learning

%Question 1:   Logistic Regression




disp('Question1: Logistic Regression');    %Displaying the header line for question 1




% a. Loading and preparing the data from the file
data = load('hw3_data1.txt');  %Loading the data from the file 
X = data(:, 1:2); %columns 1 and 2for the scores  
y = data(:, 3);  %column 3 for the admission decision

% Adding intercept term for the bias
X = [ones(size(X,1), 1) X];

% Printing the sizes
fprintf('Size of feature matrix X: %d x %d\n', size(X));
fprintf('Size of label vector y: %d x %d\n', size(y));

% b. Plotting the training data
figure;  %new window for the plot
pos = find(y == 1);  % + when y = 1
neg = find(y == 0); % o when y = 0 
plot(X(pos,2), X(pos,3), 'k+', 'LineWidth', 2, 'MarkerSize', 7);  %plotting the +
hold on;
plot(X(neg,2), X(neg,3), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);  %plotting the o
xlabel('Exam 1 score');  %The title of x axis
ylabel('Exam 2 score');   %The title of y axis
legend('Admitted', 'Not admitted'); % + Admitted and o not admitted
saveas(gcf, 'ps3-1-b.png');  %Saving the image

% c. Splitting the data into training and test sets
rng(42); % For reproducibility
train_ratio = 0.9; %Specifying that 90% of the data will be used for training
n = size(X, 1);  %Getting the total number of examples in the dataset
train_idx = randperm(n, round(train_ratio * n));  % Calculating how many examples we need for training (90% of n), Rounding to nearest integer since we can't have fractional examples, and this gives us random indices for the training set
test_idx = setdiff(1:n, train_idx);  %This gives me the remaining indices for the test set

X_train = X(train_idx, :); %Training features
y_train = y(train_idx); %Training labels
X_test = X(test_idx, :); %Test features
y_test = y(test_idx); %Test labels

% d. Testing the sigmoid function
z = -15:0.01:15;  %defining z
gz = sigmoid(z);  %Calling the sigmoid fuction
figure;  %new window for our figure
plot(z, gz, 'LineWidth', 2);  %Plotting gz versus z 
grid on;  %grid lines
xlabel('z');  %The title of axis x
ylabel('g(z)');  %The title of axis y
title('Sigmoid Function');  %The title of the plot
saveas(gcf, 'ps3-1-c.png');  %Saving the image

% e. Testing the cost function with toy dataset
X_toy = [1 1 0; 1 1 3; 1 3 1; 1 3 4];  %Defining the x toy
y_toy = [0; 1; 0; 1];  %Defining the y toy
theta_toy = [1; 0.5; 0.2];  %Defining the theta toy
[J_toy, grad_toy] = costFunction(theta_toy, X_toy, y_toy);  %Calling the cost function
fprintf('Cost for toy dataset: %f\n', J_toy);  %Printing the cost for the toy data set

% f. Training the logistic regression model
options = optimset('MaxIter', 400, 'MaxFunEvals', 400); %Setting options for fmin
initial_theta = zeros(size(X_train, 2), 1); %Initializing theta by using zeros

% Creating an anonymous function that captures X_train and y_train
costFunctionHandle = @(t) costFunction(t, X_train, y_train);
[theta, J_val] = fminsearch(costFunctionHandle, initial_theta, options);
fprintf('Optimal theta: [%f, %f, %f]\n', theta(1), theta(2), theta(3)); %Printing optimal theta
fprintf('Cost at convergence: %f\n', J_val); %Printing the cost of convergence





% g. Plotting the decision boundary
figure;  %New window
pos = find(y_train == 1); % + for y = 1
neg = find(y_train == 0); % o for y = 0
plot(X_train(pos,2), X_train(pos,3), 'k+', 'LineWidth', 2, 'MarkerSize', 7);  %Plotting +
hold on;
plot(X_train(neg,2), X_train(neg,3), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);  %Plotting o

% Plotting decision boundary
plot_x = [min(X_train(:,2))-2, max(X_train(:,2))+2];  %plot x
plot_y = (-1/theta(3))*(theta(2)*plot_x + theta(1));  %plot y
plot(plot_x, plot_y, 'b-', 'LineWidth', 2);  %Plotting the decision boundry

xlabel('Exam 1 score');  %The title of x axis
ylabel('Exam 2 score');  %The title of y axis
legend('Admitted', 'Not admitted', 'Decision Boundary');  %Using legend 
saveas(gcf, 'ps3-1-g.png'); %Saving the image

% h. Computing accuracy on test set
predictions = sigmoid(X_test * theta) >= 0.5;  %Calling the sigmoid function
accuracy = mean(predictions == y_test) * 100;  %Computing the accuracy
fprintf('Test Set Accuracy: %.2f%%\n', accuracy);  %Printing the accuracy

% i. Predict for test1=60, test2=65
new_student = [1 60 65];  %Defining the student scores
prob = sigmoid(new_student * theta);  %Calling the sigmoid function
fprintf('Admission probability for test1=60, test2=65: %.2f%%\n', prob*100);
if prob >= 0.5    %If the probabilty is greater than or equal 0.5 it will be admitted
    decision = 'Admitted';
else
   decision = 'Not admitted';  %if less than 0.5 the result is not admitted
end
fprintf('Admission decision: %s\n', decision);  %Printing the admission decision


% j. (Bonus) Starting with cost function:
%J(θ) = (1/m) * ∑[-y⁽ⁱ⁾log(h_θ(x⁽ⁱ⁾)) - (1-y⁽ⁱ⁾)log(1-h_θ(x⁽ⁱ⁾))]
% in Derivation I will Apply Chain Rule:
%I will Need to find ∂J/∂θⱼ
%I   will Consider sigmoid function h_θ(x) = g(θᵀx)
%Then I will use g'(z) = g(z)(1-g(z))
%Then I will Find derivative of h_θ(x): ∂h_θ(x)/∂θⱼ = h_θ(x)(1-h_θ(x)) * xⱼ
%For single example: I will take derivative of log terms and substitute sigmoid derivative
%Then I will simplify to: [h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾] * xⱼ⁽ⁱ⁾
%and the Final Result: ∂J/∂θⱼ = (1/m) * ∑(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾. This matches the homework's provided gradient formula

% 2. Question 2: Non-linear Fitting
disp('Question 2: Non-linear Fitting');   %Display the header for the second question

% a. Loading and preparing data
data = readmatrix('hw3_data2.csv');  %Loading the data from the file
n_p = data(:,1);  % population
profit = data(:,2);  % profit

% Creating feature matrix for quadratic fit
X = [ones(size(n_p)) n_p n_p.^2];

% Solving using normal equation
theta = pinv(X' * X) * X' * profit;
fprintf('Model parameters [θ0, θ1, θ2]: [%f, %f, %f]\n', theta(1), theta(2), theta(3));  %Printing the model parameter

% b. Plotting data and fitted model
figure;  %new window
plot(n_p, profit, 'ro', 'MarkerSize', 7);  %plotting the data
hold on;

% Plotting fitted model
n_p_plot = linspace(min(n_p), max(n_p), 100)';
X_plot = [ones(size(n_p_plot)) n_p_plot n_p_plot.^2];
profit_plot = X_plot * theta;

plot(n_p_plot, profit_plot, 'b-', 'LineWidth', 2);  %plotting
xlabel('Population in thousands, n'); %The title of x axis
ylabel('Profit');  %The title of y axis
title('Population vs. Profit with Quadratic Fit');  %The title of the plot
legend('Training data', 'Fitted model');  %using legend
grid on;   %Adding grid lines
saveas(gcf, 'ps3-2-b.png');  %Saving the image