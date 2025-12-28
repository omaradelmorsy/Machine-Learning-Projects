% kmeans_multiple - K-means with multiple random initializations
function [ids, means, ssd] = kmeans_multiple(X, K, iters, R)
    % K-means with multiple random initializations: Inputs: X: m x n data matrix (m samples, n dimensions)
    %  K: number of clusters and iters: number of iterations and  R: number of random initialization
    % Outputs: ids: m x 1 vector of best cluster assignments and means: K x n matrix of best cluster centers
    % ssd: lowest sum of squared distances found

    % Initializing with a high value
    best_ssd = Inf;
    best_ids = [];
    best_means = [];
    
    % Running K-means R times with different initializations
    for r = 1:R
        [r_ids, r_means, r_ssd] = kmeans_single(X, K, iters);
        
        % Keeping the clustering with the lowest SSD
        if r_ssd < best_ssd
            best_ssd = r_ssd;
            best_ids = r_ids;
            best_means = r_means;
        end
    end
    
    % Returning the best clustering found
    ids = best_ids;
    means = best_means;
    ssd = best_ssd;
end

