function g = sigmoid(z)
%The Sigmoid will Compute the  sigmoid function

%   g = SIGMOID(z) computes the sigmoid of z.
g = 1 ./ (1 + exp(-z));
end  %The end of the function