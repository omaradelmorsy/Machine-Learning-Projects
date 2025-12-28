%Omar Morsy

%Homework 1 
%ECE 1395 - Machine Learning

%I will answer question 1 and 2 in the report

%Question 3
%Part a   Generating random numbers of vector x from a Gaussian distribution with mean of 1.5 and standard deviation of 0.6 
x = 1.5 + 0.6 * randn(1000000, 1);

%Part b  Generating a vector z of random numbers from a uniform
%distribution [-1 3] using rand function
z = -1 + 4 * rand(1000000, 1); %I just used scales [0, 1] to [-1, 3] 

%Part c  Using the function of the histogram to plot part a and b
%I tried to use the histogram function and I got the first image  but the second one I did not get it, I do not know why so i DECIDED to use the subplot function after that and commented the histogram function part 
%Plotting the Gaussian Distribution
%figure;  %To create a figure for the histogram
%histogram(x, 'Normalization','Probability'); %Using the histogram function to plot NORMALIZED histogram of my vector x
%title('The histogram of Gaussian Distribution for vector x'); %Creating the title of the histogram
%xlabel('The Value'); %The title of axis x
%ylabel('The probability'); %The title of axis y
%saveas(gcf, './output/ps1-3-c-1.png'); %saving the image of the gaussian distribution histogram


%Plotting the Uniform Distribution
%figure;  %To create a figure for the histogram

%histogram(z, 'The Normalization','The Probability'); %Using the histogram function to plot NORMALIZED histogram of my vector z
%title('The histogram of Uniform Distribution for vector z'); %Creating the title of the histogram
%xlabel('The Value'); %The title of axis x
%ylabel('The probability'); %The title of axis y
%saveas(gcf, './output/ps1-3-c-2.png'); %saving the image of the uniform
%distribution histogram on the path i chose



%Using the subplot and histogram function
figure(1);  %Creating a figure
clf;

% First subplot - Gaussian distribution
subplot(2,1,1);  % 2 rows, 1 column, first plot. Subplot can have more than one histogram in the same window so i used it this time to make every thing easier
histogram(x, 50, 'Normalization', 'probability'); %Using the histogram function to plot NORMALIZED histogram of my vector x
title('Normalized Histogram of Gaussian Distribution'); %The title of the histogram 
xlabel('Value'); %The title of x
ylabel('Probability'); %The title of y
grid on; %to make grid lines to make more visibility

% Second subplot - Uniform distribution
subplot(2,1,2);  % 2 rows, 1 column, second plot, Subplot can have more than one histogram in the same window so i used it this time to make every thing easier
histogram(z, 50, 'Normalization', 'probability', 'FaceColor', 'b'); %Using the histogram function to plot NORMALIZED histogram of my vector z with blue color
title('Normalized Histogram of Uniform Distribution'); %The title of the histogram
xlabel('Value'); %The title of axis x
ylabel('Probability'); %The title of axis y
grid on; %for better visibility 

% Saving both plots as separate files
% Saving Gaussian plot
subplot(2,1,1); %choosing the first plot
ax1 = gca; %Getting current axis handle
fig1 = figure('Visible', 'off'); %Creating invisible figure
copyobj(ax1, fig1); %copying the subplot to the new figure 
saveas(fig1, 'images/ps1-3-c-1.png'); %saving the figure in the folder images

% Save Uniform plot
subplot(2,1,2); %choosing the second plot
ax2 = gca; %Getting current axis handle
fig2 = figure('Visible', 'off'); %Creating invisible figure 2
copyobj(ax2, fig2); %Copying the subplot to figure2
saveas(fig2, 'images/ps1-3-c-2.png'); %Saving the figure in the folder images

%Does the histogram for x look like a Gaussian distribution? Yes, Because
%it is symmetric about the mean and showing that data near the mean are
%more more frequent in occurence than data far from the mean and it looks
%like bell curve.

%Does the histogram for z look like a uniform distribution? Yes, it is very
%close to uniform distribution because it describes the form of the
%probability distribution where every possible outcome has an equal
%likelihood of happenning.



%PART d
%I will use a for loop
tic; %Starting timer
for i = 1:size(x,1) %Starting from the first index till the last index which is size
    x(i) = x(i) + 1; %increment 1 each time
end
loop_time = toc; %Stopping timer
fprintf('The time was taken by the loop: %.4f secs\n', loop_time);  %Using fprintf to print the execution time
%The execution time is 0.0154 seconds


%PART e
%Adding 1 to every value in the original random vector x without using
%loops
x = 1.5 + 0.6 * randn(1000000, 1); %I will reset the value x to the original value
tic; %starting timer 
x = x + 1; %Increment 1 each time
vector_time = toc; %to calculate the time (Stopping timer)
fprintf('The was taken without the loop: %.4f secs\n', vector_time); %Printing the execution time using fprintf
%The execution time is 0.0015 seconds. The best way to add a constant to a
%long vector is not using a loop because the execution time is 0.0015 which
%is less than the execution time with a loop which is 0.0154  that means
%matlab is faster without using a loop.


%Part f
y = z(z > 0 & z < 1.5); %defining the vector y (positive numbers in z and less than 1.5)
fprintf('The number of retrieved elements: %d\n', length(y)); %using length function to calc how many elements in y
%I reran it twice (374795 and 374605), there is a small difference because z is randomly
%generated each time and while theoretically 37.5% of values should fall
%between 0 and 1.5 



%Question 4 
%Part a
A = [2 1 3; 5 4 8; 6 3 10]; %dEFINNING THE MATRIX A
min_A_rows = min(A, [], 2); %Finding the minimum value in each row
max_A_columns = max(A, [], 1); %Finding the maximum value in each column
A_min = min(A(:)); %Finding the smallest value in A
sum_A_column = sum(A, 1); %The sum of each column
sum_A = sum(A(:)); %The sum of all elements in A
B = A.^2; %Creating matrix B


%PART b
Coeffs = [2 5 -2;2 6 4;6 8 18]; %The coefficient matrix
const = [12; 6; 15]; %the constant matrix
sol = Coeffs \ const; %getting the solution of x, y and z using that: the inverse of the coeffs matrix multiply by the constant matrix will give the solution
%The result is: x = 5.7273, y = -0.2727 and z = -0.9545


%PART c
x1 = [-5 0 2]; %Defining vector x1 
x2 = [-1 -1 0]; %Defining vector x2

L1_Norm_x1 = norm(x1, 1); %Finding the L1 norm for x1 and the result is 7 which is the same answer I got by my hand
L1_Norm_x2 = norm(x2, 1); %Finding the L1 norm for x2 and the result is 2 which is the same answer I got by my hand

L2_Norm_x1 = norm(x1, 2); %Finding the L2 norm for x1 and the result is 5.3852 which is the same answer I got by my hand
L2_Norm_x2 = norm(x2, 2); %Finding the L2 norm for x2 and the result is 1.4142 which is the same answer I got by my hand



%Question 5
%Part a
X = repmat((1:10)', 1, 3); %Creating the matrix X (10*3) using repmat
Y = ((1:10)'); %Creating the vector y (10*1)
disp('The matrix X (10*3) is:'); %Displaying or printing that line 
disp(X); %Displaying or printing the matrix X

%Part b c d
%Spliting X into submatrices X_train and X_test
%First I will shuffle the indices
for i = 1:3  %I did this loop because part d says do part b and c three times
indices = randperm(10); %Shuffling them
TrainIndices = indices(1:8); %the train indices are 8 rows
TestIndices = indices(9:10); %The test indices are 2 rows

%spliting x
X_train = X(TrainIndices, :);
X_test = X(TestIndices, :);


%PART c
%spliting Y
Y_train = X(TrainIndices);
Y_test = X(TestIndices);

%PART d
%Displaying
disp('the submatrix X_train is: '); %Displaying or printing that line
disp(X_train); %Displaying or printing X_train
disp('The submatrix X_test is: '); %Displaying or printing that line
disp(X_test); %Displaying or printing X_test

disp('The submatrix Y_train is: '); %Dislaying or printing that line
disp(Y_train); %Displaying or printing Y_train 
disp('The submatrix Y_test is: '); %Displaying or printing that line
disp(Y_test); %Displaying or printing Y_test 
%Idid not get the same submatrices for X and Y each time I split X because
% I shuffle the indices each time using randperm That is why I get different
% result every time.




end   %Ending the for-loop


