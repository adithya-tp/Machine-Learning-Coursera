function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Everything before t
% feed-forward implementation
a1 = [ones(size(X, 1),1), X];
z2 = (a1 * Theta1');
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1), a2];
z3 = (a2 * Theta2');
a3 = sigmoid(z3);

% to convert y to one hot encoded version.
y_ohe = eye(max(y))(y,:);

% compute cost.
J = (-1/m) * sum(sum(y_ohe .* log(a3) + (1 - y_ohe) .* (log(1 - a3))));

len_theta = size(Theta1, 2);  
theta_sq = Theta1(:, 2:len_theta).^2;
J = J + (lambda / (2*m)) * sum(sum(theta_sq));

len_theta = size(Theta2, 2);
theta_sq = Theta2(:, 2:len_theta).^2;
J = J + (lambda / (2*m)) * sum(sum(theta_sq));

grad1 = 0;
grad2 = 0;

for i=1:m
  % append 1 for bias term, then forward pass
  a1 = [1, X(i, :)]';
  z2 = Theta1 * a1;
  % find activation of hidden vector, then add 1 for bias term.
  a2 = sigmoid(z2);
  a2 = [1; a2];
  % keep moving forward
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  
  % get one hot encoded ground truth vector.
  y_i = y_ohe(i,:)';
  
  % compute error terms for each layer.
  delta3 = a3 - y_i;
  delta2 = (Theta2(:, 2:end)' * delta3) .* sigmoidGradient(z2);
  
  % accumulate gradient across training samples
  grad2 += delta3 * a2';
  grad1 += delta2 * a1';
end

Theta2_grad = grad2 / m;
Theta1_grad = grad1 / m;

Theta2_grad(:, 2:end) += (lambda/m) * Theta2(:, 2:end);
Theta1_grad(:, 2:end) += (lambda/m) * Theta1(:, 2:end);

%Theta2 += Theta2_grad;
%Theta1 += Theta1_grad;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
