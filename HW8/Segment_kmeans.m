function [im_out] = Segment_kmeans(im_in, K, iters, R)
    % Image segmentation using K-means: Inputs: im_in: Input image (HxWx3) and K: Number of clusters
    %  iters: Number of iterations for K-means and R: Number of random initializations
    % Outputs: im_out: Segmented image with pixels recolored according to cluster means

    % Converting the image to double
    im_in = im2double(im_in);
    
    % Getting the image dimensions
    [H, W, ~] = size(im_in);
    
    % Reshaping the image into feature matrix (each pixel is a sample with 3 features: R,G,B)
    X = reshape(im_in, H*W, 3);
    
    % Performing K-means clustering
    [ids, means, ~] = kmeans_multiple(X, K, iters, R);
    
    % Recoloring each pixel according to its cluster mean
    im_recolored = zeros(size(X));
    for k = 1:K
        % Assigning cluster mean color to all pixels in this cluster
        im_recolored(ids == k, :) = repmat(means(k, :), sum(ids == k), 1);
    end
    
    % Reshaping back to image format
    im_reshaped = reshape(im_recolored, H, W, 3);
    
    % Converting to uint8 for display
    im_out = uint8(im_reshaped * 255);
end