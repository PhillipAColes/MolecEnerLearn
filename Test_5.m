% This is a script to generate the NN input params saved in Test-5.mat
clear all;  clc
a = -0.5;
b = 0.5;
r = (b-a).*rand(1000,1) + a
 x1 = linspace(1,10,1000)' + r(randperm(length(r)));
 x2 = linspace(2,20,1000)' + r(randperm(length(r)));
 x3 = linspace(1,3,1000)' + r(randperm(length(r)))/5;
 X = [x1 x2 x3]
 y = x2 - (x1/2) + x3.^2 + x2.*x3./x1 + r
 %plot3(X(:,1),X(:,2),y,'o')
 num_hidden_layers = 1 %one hidden layer
 num_hidden_units = [ 7 ]; %seven units
 activation_function_type = { 'sigmoid', 'linear'};
 feature_scaling_tf = true;
 lambda = 0.003
 num_units = [ size(X,2) num_hidden_units 1 ];
 epsilon_init = 0.12;
 % Initialise weights array
 for layer = 1:num_hidden_layers+1   
        % num_units(layer)+1 <=== +1 comes from bias term 
     weights_array(layer) = ...
     {rand(num_units(layer)+1,num_units(layer+1)) * 2 * epsilon_init - epsilon_init};
 end
 save('Test_5','X','y','num_hidden_layers','num_hidden_units','activation_function_type',...
      'feature_scaling_tf','lambda','num_units','weights_array')