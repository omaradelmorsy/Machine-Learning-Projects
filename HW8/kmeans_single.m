% kmeans_single - Basic K-means implementation
function [ids, means, ssd] = kmeans_single(X, K, iters)
    % Implementing K-means clustering algorithm: Inputs: X: m x n data matrix (m samples, n dimensions and K: number of clusters and iters: number of iterations
    % Outputs: ids: m x 1 vector of cluster assignments (1 to K) and means: K x n matrix of cluster centers
    % ssd: sum of squared distances to respective centers

    % Geting the data dimensions
    [m, n] = size(X);
    
    % Initializing the cluster means randomly within data range
    min_vals = min(X, [], 1);
    max_vals = max(X, [], 1);
    ranges = max_vals - min_vals;
    
    % Generating K random means within the range of each feature
    means = zeros(K, n);
    for i = 1:n
        means(:, i) = min_vals(i) + ranges(i) * rand(K, 1);
    end
    
    % Initializing the cluster assignments
    ids = ones(m, 1);
    
    % K-means iterations
    for iter = 1:iters
        % Step 1: Assigning each point to the nearest cluster
        distances = pdist2(X, means);  % Compute distances to all centers
        [~, ids] = min(distances, [], 2);  % Find closest center for each point
        
        % Step 2: Recomputing the cluster means
        old_means = means;
        for k = 1:K
            if sum(ids == k) > 0  % Ensuring the cluster is not empty
                means(k, :) = mean(X(ids == k, :), 1);
            end
        end
        
        % Checking for convergence (optional)
        if all(all(old_means == means))
            break;
        end
    end
    
    % Computing the final SSD (Sum of Squared Distances)
    ssd = 0;
    for k = 1:K
        if sum(ids == k) > 0
            cluster_points = X(ids == k, :);
            cluster_center = means(k, :);
            squared_dists = sum((cluster_points - cluster_center).^2, 2);
            ssd = ssd + sum(squared_dists);
        end
    end
end

