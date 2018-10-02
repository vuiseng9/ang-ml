function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% Train the SVM

hyperparam_space = [0.01, 0.02, 0.03, 0.1, 0.2, 0.3, 1, 2, 3];

highest_acc = 0;

for C = hyperparam_space,
  for sigma = hyperparam_space,
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    val_acc = mean(double(predictions == yval));
    fprintf('C => %f, sigma => %f, val_acc => %f \n', C, sigma, val_acc*100);
    if val_acc > highest_acc
        highest_acc = val_acc;
        best_C = C;
        best_sigma = sigma;
    endif
  end
end
fprintf('best C => %f, best sigma => %f, val_acc => %f\n', best_C, best_sigma, highest_acc);
    
C = best_C;
sigma = best_sigma;






% =========================================================================

end
