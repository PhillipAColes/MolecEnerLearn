% This is a script to generate the NN input params saved in Test-4.mat
clear all;  clc
a = -0.5;
b = 0.5;
r = (b-a).*rand(1000,1) + a
 x1 = linspace(1,10,1000)' + r(randperm(length(r)));
 x2 = linspace(2,20,1000)' + r(randperm(length(r)));
 X = [x1 x2]
 y = x2 - (x1/2) + r
 plot3(X(:,1),X(:,2),y,'o')
 num_hidden_layers = 1 %one hidden layer
 num_hidden_units = [ 4 ]; 
 activation_function_type = { 'sigmoid', 'linear'};
 feature_scaling_tf = true;
 lambda = 0.003
 num_units = [ size(X,2) num_hidden_units 1 ];
 for layer = 1:num_hidden_layers+1
     % weights array set to arrays of ones
     weights_array(layer) = {ones(num_units(layer)+1,num_units(layer+1))};
 end
 save('Test_4','X','y','num_hidden_layers','num_hidden_units','activation_function_type',...
      'feature_scaling_tf','lambda','num_units','weights_array')