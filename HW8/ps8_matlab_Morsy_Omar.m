% ECE 1395 - Spring 2025
% Introduction to Machine Learning
% Homework Assignment 8: Ensemble Learning + Clustering
%Omar Morsy

% Problem 1: Bagging and Handwritten-digits recognition
% 1.a Loading the data and displaying 25 random images

load('input/HW8_data1.mat');  

% Randomly selecting 25 images
num_samples = size(X, 1);
rand_indices = randperm(num_samples, 25);
selected_images = X(rand_indices, :);

% Displaying the images in a 5x5 grid
figure('Name', 'Random 25 Images from MNIST Dataset');
for i = 1:25
    subplot(5, 5, i);
    digit_image = reshape(selected_images(i, :), 20, 20);
    imshow(digit_image, []);
    title(num2str(y(rand_indices(i))));
end
saveas(gcf, 'output/ps8-1-a-1.png');  %Saving the image 

% 1.b Splitting the data into training and testing sets
% Randomly splitting the data into training (4500) and testing (500) sets
rand_indices = randperm(num_samples);
train_indices = rand_indices(1:4500);
test_indices = rand_indices(4501:5000);

X_train = X(train_indices, :);
y_train = y(train_indices);
X_test = X(test_indices, :);
y_test = y(test_indices);

% 1.c Applying bagging to create 5 subsets
% Creating 5 subsets with random sampling with replacement
subset_size = 1200;
X_subsets = cell(5, 1);
y_subsets = cell(5, 1);

for i = 1:5
    subset_indices = randi(length(y_train), [subset_size, 1]);
    X_subsets{i} = X_train(subset_indices, :);
    y_subsets{i} = y_train(subset_indices);
end

% 1.d Training SVM (One-vs-One) with 3rd order polynomial kernel using X1
fprintf('Training SVM classifier...\n');
svm_model = fitcecoc(X_subsets{1}, y_subsets{1}, 'Learners', templateSVM('KernelFunction', 'polynomial', 'PolynomialOrder', 3));

% Calculating the errors
svm_error_X1 = 1 - sum(predict(svm_model, X_subsets{1}) == y_subsets{1}) / length(y_subsets{1});
fprintf('SVM: Error on X1 (training): %.4f\n', svm_error_X1);

% Calculating the errors on other training subsets (X2 to X5)
svm_error_other_subsets = zeros(1, 4);
for i = 2:5
    svm_error_other_subsets(i-1) = 1 - sum(predict(svm_model, X_subsets{i}) == y_subsets{i}) / length(y_subsets{i});
    fprintf('SVM: Error on X%d: %.4f\n', i, svm_error_other_subsets(i-1));
end

% Calculating the error on test set
svm_error_test = 1 - sum(predict(svm_model, X_test) == y_test) / length(y_test);
fprintf('SVM: Error on test set: %.4f\n', svm_error_test);

% 1.e Training KNN (K=15) using X2
fprintf('\nTraining KNN classifier...\n');
knn_model = fitcknn(X_subsets{2}, y_subsets{2}, 'NumNeighbors', 15);

% Calculating the errors
knn_error_X2 = 1 - sum(predict(knn_model, X_subsets{2}) == y_subsets{2}) / length(y_subsets{2});
fprintf('KNN: Error on X2 (training): %.4f\n', knn_error_X2); %Displaying

% Calculating errors on other training subsets
knn_error_other_subsets = zeros(1, 4);
for i = [1, 3, 4, 5]
    knn_error_other_subsets(i == [1, 3, 4, 5]) = 1 - sum(predict(knn_model, X_subsets{i}) == y_subsets{i}) / length(y_subsets{i});
    fprintf('KNN: Error on X%d: %.4f\n', i, knn_error_other_subsets(i == [1, 3, 4, 5])); %Displaying
end

% Calculating the error on test set
knn_error_test = 1 - sum(predict(knn_model, X_test) == y_test) / length(y_test);
fprintf('KNN: Error on test set: %.4f\n', knn_error_test); %Displaying

% 1.f Training Logistic Regression using X3
fprintf('\nTraining Logistic Regression classifier...\n');
% Using fitcecoc with linear learner for multiclass logistic regression
lr_model = fitcecoc(X_subsets{3}, y_subsets{3}, 'Learners', templateLinear('Learner', 'logistic'));

% Calculating errors
lr_error_X3 = 1 - sum(predict(lr_model, X_subsets{3}) == y_subsets{3}) / length(y_subsets{3});
fprintf('LR: Error on X3 (training): %.4f\n', lr_error_X3);

% Calculating errors on other training subsets
lr_error_other_subsets = zeros(1, 4);
for i = [1, 2, 4, 5]
    lr_error_other_subsets(i == [1, 2, 4, 5]) = 1 - sum(predict(lr_model, X_subsets{i}) == y_subsets{i}) / length(y_subsets{i});
    fprintf('LR: Error on X%d: %.4f\n', i, lr_error_other_subsets(i == [1, 2, 4, 5]));
end

% Calculating error on test set
lr_error_test = 1 - sum(predict(lr_model, X_test) == y_test) / length(y_test);
fprintf('LR: Error on test set: %.4f\n', lr_error_test);

% 1.g Training Decision Tree using X4
fprintf('\nTraining Decision Tree classifier...\n');
dt_model = fitctree(X_subsets{4}, y_subsets{4});

% Calculating errors
dt_error_X4 = 1 - sum(predict(dt_model, X_subsets{4}) == y_subsets{4}) / length(y_subsets{4});
fprintf('DT: Error on X4 (training): %.4f\n', dt_error_X4);

% Calculating the errors on other training subsets
dt_error_other_subsets = zeros(1, 4);
for i = [1, 2, 3, 5]
    dt_error_other_subsets(i == [1, 2, 3, 5]) = 1 - sum(predict(dt_model, X_subsets{i}) == y_subsets{i}) / length(y_subsets{i});
    fprintf('DT: Error on X%d: %.4f\n', i, dt_error_other_subsets(i == [1, 2, 3, 5]));
end

% Calculating error on test set
dt_error_test = 1 - sum(predict(dt_model, X_test) == y_test) / length(y_test);
fprintf('DT: Error on test set: %.4f\n', dt_error_test);

% 1.h Training Random Forest (25 trees) using X5
fprintf('\nTraining Random Forest classifier...\n');
rf_model = TreeBagger(25, X_subsets{5}, y_subsets{5});

% Calculating the errors
rf_preds_X5 = cellfun(@str2num, predict(rf_model, X_subsets{5}));
rf_error_X5 = 1 - sum(rf_preds_X5 == y_subsets{5}) / length(y_subsets{5});
fprintf('RF: Error on X5 (training): %.4f\n', rf_error_X5);

% Calculating the errors on other training subsets
rf_error_other_subsets = zeros(1, 4);
for i = 1:4
    rf_preds = cellfun(@str2num, predict(rf_model, X_subsets{i}));
    rf_error_other_subsets(i) = 1 - sum(rf_preds == y_subsets{i}) / length(y_subsets{i});
    fprintf('RF: Error on X%d: %.4f\n', i, rf_error_other_subsets(i));
end

% Calculating the error on test set
rf_preds_test = cellfun(@str2num, predict(rf_model, X_test));
rf_error_test = 1 - sum(rf_preds_test == y_test) / length(y_test);
fprintf('RF: Error on test set: %.4f\n', rf_error_test);

% 1.i Majority voting
fprintf('\nEnsemble with Majority Voting\n');
% Getting predictions from all classifiers on test set
svm_preds = predict(svm_model, X_test);
knn_preds = predict(knn_model, X_test);
lr_preds = predict(lr_model, X_test);
dt_preds = predict(dt_model, X_test);
rf_preds = cellfun(@str2num, predict(rf_model, X_test));

% Performing majority voting
ensemble_preds = zeros(size(y_test));
for i = 1:length(y_test)
    % Collecting all predictions for the current sample
    all_preds = [svm_preds(i), knn_preds(i), lr_preds(i), dt_preds(i), rf_preds(i)];
    
    % Finding the most common prediction (mode)
    ensemble_preds(i) = mode(all_preds);
end

% Calculateing error rate
ensemble_error = 1 - sum(ensemble_preds == y_test) / length(y_test);
fprintf('Ensemble Majority Voting Error on test set: %.4f\n', ensemble_error);

% Problem 2: K-means clustering and image segmentation

% 2.c Applying K-means for image segmentation
% Loading and resizing the images
image_files = dir('input/*.png'); % Adjusting path as needed

for img_idx = 1:length(image_files)
    img_path = fullfile('input', image_files(img_idx).name);
    img = imread(img_path);
    img = imresize(img, [100, 100]);  % Resizing for faster processing
    
    % Trying different combinations of parameters
    K_values = [3, 5, 7];
    iter_values = [7, 15, 30];
    R_values = [5, 8, 10];
    
    % Selecting some parameter combinations to save time
    for k_idx = 1:length(K_values)
        for iter_idx = 1:length(iter_values)
            for r_idx = 1:length(R_values)
                % Not all combinations needed
                if iter_idx > 1 && r_idx > 1
                    continue;
                end
                
                K = K_values(k_idx);
                iters = iter_values(iter_idx);
                R = R_values(r_idx);
                
                % Segment the image
                segmented_img = Segment_kmeans(img, K, iters, R);
                
                % Saving the result
                output_filename = sprintf('output/segmented_%s_K%d_iter%d_R%d.png', ...
                    image_files(img_idx).name(1:end-4), K, iters, R);
                imwrite(segmented_img, output_filename);
                
                % Displaying one combination for demonstration
                if k_idx == 2 && iter_idx == 2 && r_idx == 1
                    figure;  %New window
                    subplot(1, 2, 1);
                    imshow(img);
                    title('Original Image');  %Title
                    
                    subplot(1, 2, 2);
                    imshow(segmented_img);
                    title(sprintf('Segmented (K=%d, iter=%d, R=%d)', K, iters, R));  %Displaying
                    
                    saveas(gcf, sprintf('output/comparison_%s_K%d_iter%d_R%d.png', ...
                        image_files(img_idx).name(1:end-4), K, iters, R));
                end
            end
        end
    end
end

