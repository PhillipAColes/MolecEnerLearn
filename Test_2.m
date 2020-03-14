% This is a script to generate the NN input params saved in Test-1.mat
clear all;  clc
 y = 10;
 X = [1 1]
 num_hidden_layers = 1
 num_hidden_units = [ 2 ];
 activation_function_type = { 'sigmoid', 'linear'};
 feature_scaling_tf = false;
 lambda = 0
 num_units = [ size(X,2) num_hidden_units 1 ];
 for layer = 1:num_hidden_layers+1
     % weights array set to arrays of ones
     weights_array(layer) = {ones(num_units(layer)+1,num_units(layer+1))};
 end
 save('Test_2','X','y','num_hidden_layers','num_hidden_units','activation_function_type',...
      'feature_scaling_tf','lambda','num_units','weights_array')